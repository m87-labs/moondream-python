import base64
import json
import urllib.request
from io import BytesIO
from typing import Literal, Optional, Union

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
    Region,
    SamplingSettings,
    SegmentOutput,
    SpatialRef,
)
from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("moondream")


class CloudVL(VLM):
    def __init__(
        self,
        *,
        endpoint: str = "https://api.moondream.ai/v1",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model

    def encode_image(
        self, image: Union[Image.Image, EncodedImage]
    ) -> Base64EncodedImage:
        if isinstance(image, EncodedImage):
            assert type(image) == Base64EncodedImage
            return image
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return Base64EncodedImage(image_url=f"data:image/jpeg;base64,{img_str}")
        except Exception as e:
            raise ValueError("Failed to convert image to JPEG.") from e

    def _stream_response(self, req):
        """Helper function to stream response chunks from the API."""
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "chunk" in data:
                            yield data["chunk"]
                        if data.get("completed"):
                            break
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            "Failed to parse JSON response from server."
                        ) from e

    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short", "long"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> CaptionOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "length": length,
            "stream": stream,
        }
        if self.model is not None:
            payload["model"] = self.model
        if settings is not None:
            payload["settings"] = settings

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.endpoint}/caption",
            data=data,
            headers=headers,
        )

        def generator():
            for chunk in self._stream_response(req):
                yield chunk

        if stream:
            return {"caption": generator()}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"caption": result["caption"]}

    def query(
        self,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        spatial_refs: Optional[list[SpatialRef]] = None,
    ) -> QueryOutput:
        if question is None:
            raise ValueError("question parameter is required")

        payload = {
            "question": question,
            "stream": stream,
        }

        if image is not None:
            encoded_image = self.encode_image(image)
            payload["image_url"] = encoded_image.image_url
        if self.model is not None:
            payload["model"] = self.model
        if settings is not None:
            payload["settings"] = settings
        if reasoning:
            payload["reasoning"] = reasoning
        if spatial_refs is not None:
            payload["spatial_refs"] = spatial_refs

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.endpoint}/query",
            data=data,
            headers=headers,
        )

        if stream:
            return {"answer": self._stream_response(req)}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            output = {"answer": result["answer"]}
            if "reasoning" in result and result["reasoning"] is not None:
                output["reasoning"] = result["reasoning"]
            return output

    def chat(
        self,
        messages: list[ChatMessage],
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
    ) -> ChatOutput:
        payload = {
            "model": self.model or "moondream3-preview",
            "messages": messages,
            "stream": stream,
            "reasoning": reasoning,
        }
        if settings is not None:
            if "temperature" in settings:
                payload["temperature"] = settings["temperature"]
            if "top_p" in settings:
                payload["top_p"] = settings["top_p"]
            if "max_tokens" in settings:
                payload["max_completion_tokens"] = settings["max_tokens"]

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )

        if stream:
            return {"message": self._stream_chat_response(req)}

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
        choice = result["choices"][0]
        return {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason", "stop"),
        }

    def _stream_chat_response(self, req):
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ValueError("Failed to parse chat stream response.") from exc
                choices = event.get("choices", [])
                if not choices:
                    continue
                chunk = choices[0].get("delta", {}).get("content")
                if chunk:
                    yield chunk

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
    ) -> DetectOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "object": object,
        }
        if self.model is not None:
            payload["model"] = self.model
        if settings is not None:
            payload["settings"] = settings

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.endpoint}/detect",
            data=data,
            headers=headers,
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"objects": result["objects"]}

    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
        spatial_refs: Optional[list[SpatialRef]] = None,
    ) -> PointOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "object": object,
        }
        if spatial_refs is not None:
            payload["spatial_refs"] = spatial_refs
        if self.model is not None:
            payload["model"] = self.model
        if settings is not None:
            payload["settings"] = settings

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.endpoint}/point",
            data=data,
            headers=headers,
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {"points": result["points"]}

    def _stream_segment_response(self, req):
        """Stream segmentation response, yielding update dicts.

        The streaming format sends:
        - {"type": "bbox", "bbox": {...}} - bounding box (first message)
        - {"type": "path_delta", "chunk": "...", "completed": false} - coarse path chunks
        - {"type": "final", "path": "...", "bbox": {...}, "completed": true} - final refined path

        Yields dicts with:
        - {"bbox": Region} - when bbox is received
        - {"chunk": str} - for each coarse path chunk
        - {"path": str, "bbox": Region, "completed": True} - final message with refined path
        """
        with urllib.request.urlopen(req) as response:
            for line in response:
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        msg_type = data.get("type", "")

                        if msg_type == "bbox":
                            yield {"bbox": data.get("bbox")}
                        elif msg_type == "path_delta":
                            chunk = data.get("chunk", "")
                            if chunk:
                                yield {"chunk": chunk}
                        elif msg_type == "final":
                            yield {
                                "path": data.get("path", ""),
                                "bbox": data.get("bbox"),
                                "completed": True,
                            }
                            break
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            "Failed to parse JSON response from server."
                        ) from e

    def segment(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        spatial_refs: Optional[list[SpatialRef]] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ):
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "object": object,
            "stream": stream,
        }
        if self.model is not None:
            payload["model"] = self.model
        if spatial_refs is not None:
            payload["spatial_refs"] = spatial_refs
        if settings is not None:
            payload["settings"] = settings

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"moondream-python/{__version__}",
        }
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        req = urllib.request.Request(
            f"{self.endpoint}/segment",
            data=data,
            headers=headers,
        )

        if stream:
            return self._stream_segment_response(req)

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            output: SegmentOutput = {"path": result["path"]}
            if result.get("bbox"):
                output["bbox"] = result["bbox"]
            return output
