"""Local GPU inference backend using kestrel (Photon)."""

import asyncio
import atexit
import base64
import json
import os
import queue
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from io import BytesIO
from typing import Any, Generator, Iterator, List, Literal, Mapping, Optional, Union

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
    "moondream3-preview/01HXYZ@1000" -> ("moondream3-preview", "01HXYZ@1000")
    "Qwen/Qwen3.5-4B" -> ("Qwen/Qwen3.5-4B", None)
    """
    base, separator, suffix = model.rpartition("/")
    finetune_id, checkpoint_separator, step = suffix.rpartition("@")
    if separator and checkpoint_separator and finetune_id and step.isdigit():
        return base, suffix
    return model, None


def _build_settings(
    settings: Optional[Mapping[str, object]] = None,
    adapter: Optional[str] = None,
) -> Optional[dict]:
    """Map moondream SamplingSettings + adapter to kestrel settings dict."""
    out: dict = dict(settings or {})
    if adapter is not None:
        out["adapter"] = adapter
    return out if out else None


def _public_output(value: Any) -> Any:
    output = getattr(value, "output", None)
    return dict(output) if isinstance(output, dict) else value


def _close_unsubmitted(awaitable: Any) -> None:
    closer = getattr(awaitable, "close", None)
    if callable(closer):
        closer()


def _submit_to_loop(
    awaitable: Any,
    loop: asyncio.AbstractEventLoop,
    unavailable: str,
):
    started = threading.Event()

    async def tracked():
        started.set()
        return await awaitable

    runner = tracked()
    if not loop.is_running():
        runner.close()
        _close_unsubmitted(awaitable)
        raise RuntimeError(unavailable)
    try:
        pending = asyncio.run_coroutine_threadsafe(runner, loop)
    except RuntimeError as exc:
        runner.close()
        _close_unsubmitted(awaitable)
        raise RuntimeError(unavailable) from exc
    return pending, started, runner


def _abandon_unstarted(
    pending: Any,
    started: threading.Event,
    runner: Any,
    awaitable: Any,
) -> None:
    pending.cancel()
    if not started.is_set():
        runner.close()
        _close_unsubmitted(awaitable)


def _wait_on_loop(
    awaitable: Any,
    loop: asyncio.AbstractEventLoop,
    unavailable: str,
) -> Any:
    pending, started, runner = _submit_to_loop(awaitable, loop, unavailable)
    while True:
        try:
            return pending.result(timeout=0.05)
        except FutureTimeoutError:
            if pending.done():
                return pending.result()
            if not loop.is_running():
                _abandon_unstarted(pending, started, runner, awaitable)
                raise RuntimeError(unavailable)


async def _await_on_loop(
    awaitable: Any,
    loop: asyncio.AbstractEventLoop,
    unavailable: str,
) -> Any:
    pending, started, runner = _submit_to_loop(awaitable, loop, unavailable)
    wrapped = asyncio.wrap_future(pending)
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(wrapped),
                    timeout=0.05,
                )
            except asyncio.TimeoutError:
                if wrapped.done():
                    return wrapped.result()
                if not loop.is_running():
                    _abandon_unstarted(pending, started, runner, awaitable)
                    raise RuntimeError(unavailable)
    except asyncio.CancelledError:
        pending.cancel()
        raise


def _bridge_async_iterator(source: Any, source_loop: asyncio.AbstractEventLoop):
    iterator = source.__aiter__()

    async def proxy():
        try:
            while True:
                async def next_chunk():
                    return await iterator.__anext__()

                pending = asyncio.run_coroutine_threadsafe(next_chunk(), source_loop)
                try:
                    yield await asyncio.wrap_future(pending)
                except StopAsyncIteration:
                    return
        finally:
            closer = getattr(iterator, "aclose", None)
            if callable(closer):
                async def close_source():
                    await closer()

                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(close_source(), source_loop)
                )

    return proxy()


class PhotonStream(Iterator[dict[str, object]]):
    """Synchronous or asynchronous view of a Photon model stream."""

    def __init__(self, stream: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._stream = stream
        self._loop = loop
        self._finished = False

    def _wait(self, awaitable):
        return _wait_on_loop(
            awaitable,
            self._loop,
            "Photon stream is unavailable because its engine is closed",
        )

    async def _await(self, awaitable):
        return await _await_on_loop(
            awaitable,
            self._loop,
            "Photon stream is unavailable because its engine is closed",
        )

    @staticmethod
    def _update_output(update: Any) -> dict[str, object]:
        output = getattr(update, "output", None)
        if isinstance(output, dict):
            return dict(output)
        text = getattr(update, "text", None)
        if isinstance(text, str):
            return {"text": text}
        raise TypeError("Photon stream update has no public output")

    def __iter__(self) -> "PhotonStream":
        return self

    def __next__(self) -> dict[str, object]:
        if self._finished:
            raise StopIteration
        try:
            update = self._wait(self._stream.__anext__())
        except StopAsyncIteration:
            self._finished = True
            raise StopIteration from None

        return self._update_output(update)

    def __aiter__(self) -> "PhotonStream":
        return self

    async def __anext__(self) -> dict[str, object]:
        if self._finished:
            raise StopAsyncIteration
        try:
            update = await self._await(self._stream.__anext__())
        except StopAsyncIteration:
            self._finished = True
            raise
        return self._update_output(update)

    def updates(self) -> "PhotonStream":
        return self

    def send(self, **chunk: Any) -> None:
        sender = getattr(self._stream, "send", None)
        if not callable(sender):
            raise TypeError("this Photon stream does not accept input chunks")
        self._wait(sender(**chunk))

    async def asend(self, **chunk: Any) -> None:
        sender = getattr(self._stream, "send", None)
        if not callable(sender):
            raise TypeError("this Photon stream does not accept input chunks")
        await self._await(sender(**chunk))

    def result(self) -> dict[str, object]:
        result = _public_output(self._wait(self._stream.result()))
        if not isinstance(result, dict):
            raise TypeError("Photon stream returned no public result")
        return result

    async def aresult(self) -> dict[str, object]:
        result = _public_output(await self._await(self._stream.result()))
        if not isinstance(result, dict):
            raise TypeError("Photon stream returned no public result")
        return result

    def close(self) -> Optional[dict[str, object]]:
        closer = getattr(self._stream, "close", None)
        if not callable(closer):
            closer = getattr(self._stream, "aclose", None)
        if not callable(closer):
            self._finished = True
            return None
        result = _public_output(self._wait(closer()))
        self._finished = True
        return result if isinstance(result, dict) else None

    async def aclose(self) -> Optional[dict[str, object]]:
        closer = getattr(self._stream, "close", None)
        if not callable(closer):
            closer = getattr(self._stream, "aclose", None)
        if not callable(closer):
            self._finished = True
            return None
        result = _public_output(await self._await(closer()))
        self._finished = True
        return result if isinstance(result, dict) else None

    def __enter__(self) -> "PhotonStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aenter__(self) -> "PhotonStream":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()


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
    """Client for local Photon model capabilities."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
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

    def _prepare_invocation(
        self,
        task: str,
        prompt: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        if not isinstance(task, str) or not task or task.startswith("_"):
            raise ValueError("task must be a public Photon capability name")
        if not self.supports(task):
            raise ValueError(
                f"Model {self.model_id!r} does not support {task!r} "
                f"(supports: {', '.join(self.tasks) or 'none'})"
            )
        capability = getattr(self._model, task, None)
        if not callable(capability):
            raise RuntimeError(
                f"Photon advertises {task!r} but its client has no matching verb"
            )

        owned_prompt = dict(prompt)
        if self._adapter is not None:
            owned_prompt["settings"] = _build_settings(
                owned_prompt.get("settings"),
                self._adapter,
            )
        return capability, owned_prompt

    def invoke(self, task: str, /, **prompt: Any) -> Any:
        """Invoke any capability advertised by this Photon model."""
        capability, owned_prompt = self._prepare_invocation(task, prompt)
        return self._adapt_result(self._run(capability(**owned_prompt)))

    def run(self, task: str, inputs: Any) -> Any:
        """Run a task on a single-pass Photon model."""
        return _public_output(self._run(self._model.run(task, inputs)))

    def stream(self, task: str, /, **initial_prompt: Any) -> PhotonStream:
        """Open a caller-driven session on a stateful streaming Photon model."""
        stream = self._run(self._model.stream(task, **initial_prompt))
        return PhotonStream(stream, self._loop)

    def transcribe(self, **prompt: Any) -> Union[dict[str, object], PhotonStream]:
        """Transcribe or translate audio with a speech-capable Photon model."""
        owned_prompt = dict(prompt)
        audio = owned_prompt.get("audio")
        if callable(getattr(audio, "__aiter__", None)):
            try:
                source_loop = asyncio.get_running_loop()
            except RuntimeError:
                source_loop = None
            if source_loop is not None and source_loop is not self._loop:
                if not owned_prompt.get("stream", False):
                    raise RuntimeError(
                        "async live audio with stream=False must use "
                        "await speech.atranscribe(...)"
                    )
                owned_prompt["audio"] = _bridge_async_iterator(audio, source_loop)
        result = self.invoke("transcribe", **owned_prompt)
        if not isinstance(result, (dict, PhotonStream)):
            raise TypeError("Photon transcription returned an unsupported result")
        return result

    async def atranscribe(
        self,
        **prompt: Any,
    ) -> Union[dict[str, object], PhotonStream]:
        """Asynchronously transcribe audio without blocking the caller loop."""
        owned_prompt = dict(prompt)
        audio = owned_prompt.get("audio")
        if callable(getattr(audio, "__aiter__", None)):
            source_loop = asyncio.get_running_loop()
            if source_loop is not self._loop:
                owned_prompt["audio"] = _bridge_async_iterator(audio, source_loop)
        capability, owned_prompt = self._prepare_invocation(
            "transcribe",
            owned_prompt,
        )
        result = self._adapt_result(
            await self._arun(capability(**owned_prompt))
        )
        if not isinstance(result, (dict, PhotonStream)):
            raise TypeError("Photon transcription returned an unsupported result")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, coro):
        """Run an async coroutine on the background loop and return result."""
        return _wait_on_loop(coro, self._loop, "Photon client is closed")

    async def _arun(self, coro):
        """Run a coroutine on the Photon loop without blocking the caller loop."""
        return await _await_on_loop(coro, self._loop, "Photon client is closed")

    def _adapt_result(self, value: Any) -> Any:
        if callable(getattr(value, "__anext__", None)) and callable(
            getattr(value, "result", None)
        ):
            return PhotonStream(value, self._loop)
        return _public_output(value)

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
