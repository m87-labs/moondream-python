import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from moondream.photon_client import PhotonClient
from moondream.types import Base64EncodedImage


def client_for(model, adapter=None):
    client = object.__new__(PhotonClient)
    client._adapter = adapter
    client._model = model
    client._run = asyncio.run
    return client


class VisionModel:
    model_id = "vision-model"
    tasks = ("detect", "embed")

    def supports(self, task):
        return task in self.tasks


def test_embed_returns_owned_output_mapping_without_converting_tensors():
    tokens = torch.zeros(1, 257, 384)
    output = {"last_hidden_state": tokens, "pooler_output": tokens[:, 0]}

    class Model(VisionModel):
        async def embed(self, *, image):
            assert image == b"encoded image"
            return SimpleNamespace(output=output)

    image = Base64EncodedImage(
        image_url="data:image/png;base64," + base64.b64encode(b"encoded image").decode()
    )
    result = client_for(Model()).embed(image)

    assert result == output
    assert result is not output
    assert result["last_hidden_state"] is tokens
    assert result["pooler_output"] is output["pooler_output"]


def test_embed_preserves_pil_images_without_a_lossy_round_trip():
    source = Image.new("RGB", (8, 8))

    class Model(VisionModel):
        async def embed(self, *, image):
            assert image is source
            return SimpleNamespace(output={"features": torch.ones(1, 2)})

    result = client_for(Model()).embed(source)
    assert result["features"].shape == (1, 2)


def test_detect_image_only_omits_object_and_settings_and_preserves_labels():
    source = Image.new("RGB", (8, 8))
    objects = [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4,
                "score": 0.9, "class_id": 1, "label": "person"}]

    class Model(VisionModel):
        async def detect(self, *, image):
            assert image is source
            return SimpleNamespace(output={"objects": objects})

    assert client_for(Model()).detect(source) == {"objects": objects}


def test_detect_image_only_decodes_encoded_images_without_reencoding():
    source_bytes = b"encoded image"
    source = Base64EncodedImage(
        image_url="data:image/png;base64," + base64.b64encode(source_bytes).decode()
    )

    class Model(VisionModel):
        async def detect(self, *, image):
            assert image == source_bytes
            return SimpleNamespace(output={"objects": []})

    assert client_for(Model()).detect(source) == {"objects": []}


def test_detect_forwards_threshold_zero_and_result_limit_as_top_level_options():
    class Model(VisionModel):
        async def detect(self, *, image, threshold, max_objects):
            assert threshold == 0.0
            assert max_objects == 1
            return SimpleNamespace(output={"objects": []})

    result = client_for(Model()).detect(
        Image.new("RGB", (8, 8)), threshold=0.0, max_objects=1
    )
    assert result == {"objects": []}


@pytest.mark.parametrize("adapter", [None, "ft_example@1000"])
def test_detect_preserves_positional_object_settings_and_adapter(adapter):
    settings = {"max_objects": 7, "temperature": 0.0}

    class Model(VisionModel):
        async def detect(self, *, image, object, settings):
            assert isinstance(image, bytes) and image.startswith(b"\xff\xd8")
            assert object == "car"
            assert settings["max_objects"] == 7
            assert settings.get("adapter") == adapter
            settings["max_objects"] = 1
            return SimpleNamespace(output={"objects": []})

    result = client_for(Model(), adapter).detect(
        Image.new("RGB", (8, 8)), "car", settings
    )
    assert result == {"objects": []}
    assert settings == {"max_objects": 7, "temperature": 0.0}


def test_detect_passes_explicit_empty_settings_without_injecting_defaults():
    class Model(VisionModel):
        async def detect(self, *, image, settings):
            assert settings == {}
            return SimpleNamespace(output={"objects": []})

    assert client_for(Model()).detect(Image.new("RGB", (8, 8)), settings={}) == {
        "objects": []
    }


def test_detect_does_not_silently_drop_an_unsupported_object_prompt():
    class Model(VisionModel):
        async def detect(self, *, image):
            return SimpleNamespace(output={"objects": []})

    with pytest.raises(TypeError, match="object"):
        client_for(Model()).detect(Image.new("RGB", (8, 8)), "car")


@pytest.mark.parametrize("task", ["embed", "detect"])
def test_vision_methods_reject_unadvertised_capabilities(task):
    class Model(VisionModel):
        tasks = ("query",)

    with pytest.raises(ValueError, match=f"does not support '{task}'"):
        getattr(client_for(Model()), task)(Image.new("RGB", (8, 8)))
