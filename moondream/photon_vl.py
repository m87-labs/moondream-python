"""Local GPU inference backend using kestrel (Photon)."""

import asyncio
import base64
import os
import queue
import threading
from io import BytesIO
from typing import Generator, List, Literal, Optional, Union

from PIL import Image

from .types import (
    VLM,
    Base64EncodedImage,
    CaptionOutput,
    DetectOutput,
    EncodedImage,
    PointOutput,
    QueryOutput,
    SamplingSettings,
    SegmentOutput,
    SpatialRef,
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
    """
    if "/" not in model:
        return model, None
    base, adapter = model.split("/", 1)
    return base, adapter


def _build_settings(
    settings: Optional[SamplingSettings] = None,
    adapter: Optional[str] = None,
) -> Optional[dict]:
    """Map moondream SamplingSettings + adapter to kestrel settings dict."""
    out: dict = {}
    if settings is not None:
        if "max_tokens" in settings:
            out["max_tokens"] = settings["max_tokens"]
    if adapter is not None:
        out["adapter"] = adapter
    return out if out else None


# ------------------------------------------------------------------
# Singleton engine cache
# ------------------------------------------------------------------
# Keyed by (base_model, device, max_batch_size, kv_cache_pages) so that
# PhotonVL instances differing only by adapter share the same engine.

_engine_cache: dict[tuple, tuple] = {}  # key -> (engine, loop, thread)
_cache_lock = threading.Lock()


def _get_or_create_engine(
    base_model: str,
    max_batch_size: int,
    kv_cache_pages: Optional[int],
    device: str,
):
    """Return a shared (engine, loop, thread) for the given config."""
    key = (base_model, device, max_batch_size, kv_cache_pages)

    with _cache_lock:
        if key in _engine_cache:
            return _engine_cache[key]

    # Import kestrel lazily so non-GPU environments can still import moondream.
    from kestrel import InferenceEngine
    from kestrel.config import RuntimeConfig

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    cfg = RuntimeConfig(
        model=base_model,
        max_batch_size=max_batch_size,
        kv_cache_pages=kv_cache_pages,
        device=device,
    )

    engine = asyncio.run_coroutine_threadsafe(
        InferenceEngine.create(cfg), loop
    ).result()

    entry = (engine, loop, thread)
    with _cache_lock:
        # Another thread may have raced us; use the winner.
        if key in _engine_cache:
            # Shut down the engine we just created.
            asyncio.run_coroutine_threadsafe(engine.shutdown(), loop).result()
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            return _engine_cache[key]
        _engine_cache[key] = entry

    return entry


class PhotonVL(VLM):
    """Local GPU inference via kestrel's InferenceEngine."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "moondream3-preview",
        max_batch_size: int = 4,
        kv_cache_pages: Optional[int] = None,
        device: str = "cuda",
    ):
        if api_key is not None:
            os.environ["MOONDREAM_API_KEY"] = api_key

        base_model, self._adapter = _parse_model(model)
        self._engine, self._loop, self._thread = _get_or_create_engine(
            base_model, max_batch_size, kv_cache_pages, device
        )

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
                self._engine.caption(
                    image_bytes,
                    length=length,
                    stream=True,
                    settings=engine_settings,
                )
            )
            return {"caption": gen}

        result = self._run(
            self._engine.caption(
                image_bytes,
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
    ) -> QueryOutput:
        if question is None:
            raise ValueError("question parameter is required")

        image_bytes = _image_to_bytes(image) if image is not None else None
        engine_settings = self._settings(settings)

        if stream:
            gen = self._stream_to_generator(
                self._engine.query(
                    image=image_bytes,
                    question=question,
                    reasoning=reasoning,
                    stream=True,
                    settings=engine_settings,
                )
            )
            return {"answer": gen}

        result = self._run(
            self._engine.query(
                image=image_bytes,
                question=question,
                reasoning=reasoning,
                stream=False,
                settings=engine_settings,
            )
        )
        output: QueryOutput = {"answer": result.output["answer"]}
        if "reasoning" in result.output and result.output["reasoning"] is not None:
            output["reasoning"] = result.output["reasoning"]
        return output

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
    ) -> DetectOutput:
        image_bytes = _image_to_bytes(image)
        result = self._run(
            self._engine.detect(image_bytes, object, settings=self._settings(settings))
        )
        return {"objects": result.output["objects"]}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
    ) -> PointOutput:
        image_bytes = _image_to_bytes(image)
        result = self._run(
            self._engine.point(image_bytes, object, settings=self._settings(settings))
        )
        return {"points": result.output["points"]}

    def segment(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        spatial_refs: Optional[List[SpatialRef]] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> SegmentOutput:
        image_bytes = _image_to_bytes(image)
        result = self._run(
            self._engine.segment(
                image_bytes,
                object,
                spatial_refs=spatial_refs,
                settings=self._settings(settings),
            )
        )
        seg = result.output["segments"][0]
        output: SegmentOutput = {"path": seg["path"]}
        if seg.get("bbox"):
            output["bbox"] = seg["bbox"]
        return output
