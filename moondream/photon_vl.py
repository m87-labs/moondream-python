"""Local GPU inference backend using kestrel (Photon)."""

import asyncio
import atexit
import base64
import json
import os
import queue
import threading
from io import BytesIO
from typing import Generator, List, Literal, Optional, Union

import torch
from PIL import Image

from .types import (
    VLM,
    Base64EncodedImage,
    CaptionOutput,
    ChatMessage,
    ChatOutput,
    DetectOutput,
    EncodedImage,
    PointOutput,
    QueryOutput,
    SamplingSettings,
    SegmentOutput,
    SegmentStreamOutput,
    SpatialRef,
)


def _default_photon_device() -> str:
    """Choose the local Photon device when the caller does not specify one."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.cuda.is_built():
        # CUDA was installed but failed to initialize. Let Kestrel validate the
        # explicit CUDA device so users get the same driver/runtime diagnostic
        # as they would when passing device="cuda" themselves.
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "Photon local inference needs a supported accelerator, but neither "
        "CUDA nor Apple Silicon MPS is available in this Python environment."
    )


def _image_to_bytes(image: Union[Image.Image, EncodedImage]) -> bytes:
    """Convert a PIL Image or Base64EncodedImage to raw JPEG bytes."""
    if isinstance(image, Base64EncodedImage):
        # Strip data URI prefix if present
        data = image.image_url
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

    if isinstance(image, EncodedImage):
        raise ValueError(f"Unsupported EncodedImage type: {type(image)}")

    # PIL Image
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _parse_model(model: str) -> tuple[str, Optional[str]]:
    """Parse a model string into (base_model, adapter).

    "moondream3-preview" -> ("moondream3-preview", None)
    "moondream3-preview/ft_abc@1000" -> ("moondream3-preview", "ft_abc@1000")
    "Qwen/Qwen3.5-4B" -> ("Qwen/Qwen3.5-4B", None)
    """
    base, separator, suffix = model.rpartition("/")
    if separator and suffix.startswith("ft_"):
        return base, suffix
    return model, None


def _build_settings(
    settings: Optional[SamplingSettings] = None,
    adapter: Optional[str] = None,
) -> Optional[dict]:
    """Map moondream SamplingSettings + adapter to kestrel settings dict."""
    out: dict = dict(settings or {})
    if adapter is not None:
        out["adapter"] = adapter
    return out if out else None


# ------------------------------------------------------------------
# Singleton engine cache
# ------------------------------------------------------------------
# PhotonVL instances differing only by adapter share an engine. Credentials
# remain isolated because the engine owns the adapter provider for its key.

_engine_cache: dict[tuple, tuple] = {}  # key -> (engine, loop, thread, refs)
_cache_lock = threading.Lock()


def _stop_engine(engine, loop, thread, *, suppress_errors: bool = False) -> None:
    failure = None
    if loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(
                engine.shutdown(), loop
            ).result(timeout=30)
        except Exception as exc:
            failure = exc
        finally:
            loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=30)
    if failure is not None and not suppress_errors:
        raise failure


def _shutdown_cached_engines() -> None:
    """Stop shared engines before Python tears down CUDA and worker threads."""
    with _cache_lock:
        entries = list(_engine_cache.values())
        _engine_cache.clear()

    for engine, loop, thread, _refs in entries:
        _stop_engine(engine, loop, thread, suppress_errors=True)


# Engine shutdown may perform async DNS during its final telemetry flush.
# CPython tears down the shared thread-pool executor before normal atexit
# callbacks, so clean up before thread shutdown when that hook is available.
_register_shutdown = getattr(threading, "_register_atexit", atexit.register)
_register_shutdown(_shutdown_cached_engines)


def _get_or_create_engine(
    base_model: str,
    runtime_config: dict,
    api_key: Optional[str] = None,
):
    """Acquire a shared engine for the given config."""
    effective_api_key = (
        api_key if api_key is not None else os.environ.get("MOONDREAM_API_KEY")
    )
    key = (base_model, tuple(sorted(runtime_config.items())), effective_api_key)

    with _cache_lock:
        if key in _engine_cache:
            engine, loop, thread, refs = _engine_cache[key]
            _engine_cache[key] = (engine, loop, thread, refs + 1)
            return engine, loop, thread, key

    # Import kestrel lazily so non-GPU environments can still import moondream.
    from kestrel import InferenceEngine
    from kestrel.config import RuntimeConfig

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    try:
        cfg = RuntimeConfig(
            model=base_model,
            **runtime_config,
        )

        engine = asyncio.run_coroutine_threadsafe(
            InferenceEngine.create(cfg, api_key=effective_api_key), loop
        ).result()
    except Exception:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        raise

    loser = None
    with _cache_lock:
        # Another thread may have raced us; use the winner.
        if key in _engine_cache:
            winner_engine, winner_loop, winner_thread, refs = _engine_cache[key]
            _engine_cache[key] = (
                winner_engine,
                winner_loop,
                winner_thread,
                refs + 1,
            )
            loser = (engine, loop, thread)
        else:
            _engine_cache[key] = (engine, loop, thread, 1)

    if loser is not None:
        try:
            _stop_engine(*loser)
        except Exception:
            _release_engine(key)
            raise
        return winner_engine, winner_loop, winner_thread, key

    return engine, loop, thread, key


def _release_engine(key: tuple) -> None:
    entry = None
    with _cache_lock:
        cached = _engine_cache.get(key)
        if cached is None:
            return
        engine, loop, thread, refs = cached
        if refs > 1:
            _engine_cache[key] = (engine, loop, thread, refs - 1)
            return
        entry = _engine_cache.pop(key)

    _stop_engine(*entry[:3])


class PhotonVL(VLM):
    """Local GPU inference via kestrel's InferenceEngine."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "moondream3-preview",
        **runtime_config,
    ):
        base_model, self._adapter = _parse_model(model)
        if runtime_config.get("device") is None:
            runtime_config["device"] = _default_photon_device()
        (
            self._engine,
            self._loop,
            self._thread,
            self._engine_key,
        ) = _get_or_create_engine(base_model, runtime_config, api_key=api_key)
        self._model = self._engine.model()

    def close(self) -> None:
        """Release this client's reference to its shared local engine."""
        with _cache_lock:
            key = self._engine_key
            if key is None:
                return
            self._engine_key = None
        _release_engine(key)

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def tasks(self) -> tuple[str, ...]:
        return self._model.tasks

    def supports(self, task: str) -> bool:
        return self._model.supports(task)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        """Run an async coroutine on the background loop and return result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _stream_to_generator(self, coro) -> Generator[str, None, None]:
        """Bridge an async EngineStream into a sync generator of text chunks."""
        q: queue.Queue = queue.Queue()

        async def _consume():
            try:
                stream = await coro
                async for update in stream:
                    q.put(update.text)
                q.put(None)  # sentinel
            except Exception as exc:
                q.put(exc)

        asyncio.run_coroutine_threadsafe(_consume(), self._loop)

        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _segment_stream_to_generator(self, coro) -> SegmentStreamOutput:
        """Bridge Kestrel's segment stream into the public update shape."""
        q: queue.Queue = queue.Queue()

        async def _consume():
            try:
                stream = await coro
                async for update in stream:
                    text = update.text
                    if text.startswith("__BBOX__"):
                        q.put({"bbox": json.loads(text[len("__BBOX__"):])})
                    elif text:
                        q.put({"chunk": text})

                result = await stream.result()
                segment = result.output["segments"][0]
                q.put(
                    {
                        "path": segment["svg_path"],
                        "bbox": segment.get("bbox"),
                        "completed": True,
                    }
                )
                q.put(None)
            except Exception as exc:
                q.put(exc)

        asyncio.run_coroutine_threadsafe(_consume(), self._loop)

        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _settings(
        self, settings: Optional[SamplingSettings] = None
    ) -> Optional[dict]:
        """Build engine settings with this instance's adapter."""
        return _build_settings(settings, self._adapter)

    # ------------------------------------------------------------------
    # VLM interface
    # ------------------------------------------------------------------

    def encode_image(
        self, image: Union[Image.Image, EncodedImage]
    ) -> Base64EncodedImage:
        """Encode image to Base64EncodedImage (same as CloudVL).

        For the local backend the kestrel prefix cache handles reuse
        automatically, so this just converts to the common format.
        """
        if isinstance(image, EncodedImage):
            assert type(image) == Base64EncodedImage
            return image
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return Base64EncodedImage(image_url=f"data:image/jpeg;base64,{img_str}")

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short", "long"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        image_bytes = _image_to_bytes(image)
        engine_settings = self._settings(settings)

        if stream:
            gen = self._stream_to_generator(
                self._model.caption(
                    image=image_bytes,
                    length=length,
                    stream=True,
                    settings=engine_settings,
                )
            )
            return {"caption": gen}

        result = self._run(
            self._model.caption(
                image=image_bytes,
                length=length,
                stream=False,
                settings=engine_settings,
            )
        )
        return {"caption": result.output["caption"]}

    def query(
        self,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        spatial_refs: Optional[List[SpatialRef]] = None,
    ) -> QueryOutput:
        if question is None:
            raise ValueError("question parameter is required")

        image_bytes = _image_to_bytes(image) if image is not None else None
        engine_settings = self._settings(settings)

        if stream:
            gen = self._stream_to_generator(
                self._model.query(
                    image=image_bytes,
                    question=question,
                    reasoning=reasoning,
                    spatial_refs=spatial_refs,
                    stream=True,
                    settings=engine_settings,
                )
            )
            return {"answer": gen}

        result = self._run(
            self._model.query(
                image=image_bytes,
                question=question,
                reasoning=reasoning,
                spatial_refs=spatial_refs,
                stream=False,
                settings=engine_settings,
            )
        )
        output: QueryOutput = {"answer": result.output["answer"]}
        if "reasoning" in result.output and result.output["reasoning"] is not None:
            output["reasoning"] = result.output["reasoning"]
        return output

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        reasoning: Optional[bool] = None,
    ) -> ChatOutput:
        prompt: dict[str, object] = {
            "messages": messages,
            "stream": stream,
            "settings": self._settings(settings),
        }
        if reasoning is not None:
            prompt["reasoning"] = reasoning

        if stream:
            return {
                "message": self._stream_to_generator(
                    self._model.chat(**prompt)
                )
            }

        result = self._run(self._model.chat(**prompt))
        return {
            "message": result.output["message"],
            "finish_reason": (
                result.output.get("finish_reason")
                or result.finish_reason
                or "stop"
            ),
        }

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
    ) -> DetectOutput:
        image_bytes = _image_to_bytes(image)
        result = self._run(
            self._model.detect(
                image=image_bytes,
                object=object,
                settings=self._settings(settings),
            )
        )
        return {"objects": result.output["objects"]}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
        spatial_refs: Optional[List[SpatialRef]] = None,
    ) -> PointOutput:
        image_bytes = _image_to_bytes(image)
        result = self._run(
            self._model.point(
                image=image_bytes,
                object=object,
                settings=self._settings(settings),
                spatial_refs=spatial_refs,
            )
        )
        return {"points": result.output["points"]}

    def segment(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        spatial_refs: Optional[List[SpatialRef]] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> Union[SegmentOutput, SegmentStreamOutput]:
        image_bytes = _image_to_bytes(image)
        if stream:
            return self._segment_stream_to_generator(
                self._model.segment(
                    image=image_bytes,
                    object=object,
                    spatial_refs=spatial_refs,
                    stream=True,
                    settings=self._settings(settings),
                )
            )

        result = self._run(
            self._model.segment(
                image=image_bytes,
                object=object,
                spatial_refs=spatial_refs,
                settings=self._settings(settings),
            )
        )
        seg = result.output["segments"][0]
        output: SegmentOutput = {"path": seg["svg_path"]}
        if seg.get("bbox"):
            output["bbox"] = seg["bbox"]
        return output
