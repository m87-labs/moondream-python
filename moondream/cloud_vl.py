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
    DetectOutput,
    EncodedImage,
    PointOutput,
    QueryOutput,
    SamplingSettings,
)
from .version import __version__


class CloudVL(VLM):
    def __init__(
        self,
        *,
        endpoint: str = "https://api.moondream.ai/v1",
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint

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
        variant: Optional[str] = None,
    ) -> CaptionOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "length": length,
            "stream": stream,
        }
        if settings is not None:
            payload["settings"] = settings
        if variant is not None:
            payload["variant"] = variant

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
        variant: Optional[str] = None,
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
            
        if settings is not None:
            payload["settings"] = settings
        if reasoning:
            payload["reasoning"] = reasoning
        if variant is not None:
            payload["variant"] = variant

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

    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        settings: Optional[SamplingSettings] = None,
        variant: Optional[str] = None,
    ) -> DetectOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "object": object,
        }
        if settings is not None:
            payload["settings"] = settings
        if variant is not None:
            payload["variant"] = variant

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
        variant: Optional[str] = None,
    ) -> PointOutput:
        encoded_image = self.encode_image(image)
        payload = {
            "image_url": encoded_image.image_url,
            "object": object,
        }
        if settings is not None:
            payload["settings"] = settings
        if variant is not None:
            payload["variant"] = variant

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
