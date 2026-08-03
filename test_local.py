import sys
import os
import asyncio
from types import SimpleNamespace

import moondream as md
from PIL import Image
from moondream.photon_vl import PhotonVL, _parse_model


def test_model_parser_preserves_registered_repository_ids():
    assert _parse_model("Qwen/Qwen3.5-4B") == ("Qwen/Qwen3.5-4B", None)
    assert _parse_model("google/gemma-4-E2B-it") == (
        "google/gemma-4-E2B-it",
        None,
    )


def test_model_parser_extracts_explicit_finetune_suffix():
    assert _parse_model("moondream3-preview/01HXYZ@1000") == (
        "moondream3-preview",
        "01HXYZ@1000",
    )
    assert _parse_model("moondream3-preview/ft_abc@1000") == (
        "moondream3-preview",
        "ft_abc@1000",
    )


def test_model_parser_rejects_malformed_finetune_suffixes():
    assert _parse_model("moondream3-preview/01HXYZ") == (
        "moondream3-preview/01HXYZ",
        None,
    )
    assert _parse_model("moondream3-preview/01HXYZ@latest") == (
        "moondream3-preview/01HXYZ@latest",
        None,
    )


def test_vl_local_delegates_to_photon(monkeypatch):
    captured = {}

    def fake_photon(model, *, api_key=None, **runtime_config):
        captured.update(
            model=model,
            api_key=api_key,
            runtime_config=runtime_config,
        )
        return "photon"

    monkeypatch.setattr(md, "photon", fake_photon)

    result = md.vl(
        api_key="key",
        local=True,
        model="Qwen/Qwen3.5-4B",
        max_batch_size=8,
    )

    assert result == "photon"
    assert captured == {
        "model": "Qwen/Qwen3.5-4B",
        "api_key": "key",
        "runtime_config": {"max_batch_size": 8},
    }


def test_vl_local_requires_model():
    try:
        md.vl(local=True)
    except TypeError as exc:
        assert str(exc) == "vl(local=True) requires model=<model identifier>"
    else:
        raise AssertionError("local Photon inference accepted no model")


def test_photon_requires_model():
    try:
        md.photon()
    except TypeError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("Photon accepted no model")


def test_photon_chat_preserves_model_reasoning_default():
    calls = []

    class FakeModel:
        async def chat(self, **prompt):
            calls.append(prompt)
            return SimpleNamespace(
                output={
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": None,
                },
                finish_reason=None,
            )

    client = object.__new__(PhotonVL)
    client._adapter = None
    client._model = FakeModel()
    client._run = asyncio.run

    result = client.chat([{"role": "user", "content": "question"}])
    client.chat(
        [{"role": "user", "content": "question"}],
        reasoning=False,
    )

    assert "reasoning" not in calls[0]
    assert calls[1]["reasoning"] is False
    assert result["finish_reason"] == "stop"


def main(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Instantiate the client in local mode.
    client = md.vl(local=True, model="moondream3-preview")

    # Test the caption method.
    try:
        print("Starting caption")
        caption_output = client.caption(image, settings={"max_tokens": 10})
        print("Caption:", caption_output.get("caption"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption test failed:", e)

    try:
        print("Starting caption stream")
        for chunk in client.caption(image, length="long", stream=True)["caption"]:
            print(chunk, end="", flush=True)
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption stream test failed:", e)

    try:
        print("start Query")
        query_output = client.query(
            image, "What's in the image?", settings={"max_tokens": 10}
        )
        print("Query Answer:", query_output.get("answer"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Query test failed:", e)

    try:
        print("Starting query stream")
        for chunk in client.query(image, "What's in the image?", stream=True)["answer"]:
            print(chunk, end="", flush=True)
        print("\n------ done ------\n")
    except Exception as e:
        print("query stream test failed:", e)

    # Test the detect method.
    try:
        print("Starting output")
        detect_output = client.detect(image, "item")
        print("Detected Objects:", detect_output.get("objects"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Detect test failed:", e)

    # Test the point method.
    try:
        print("Starting Point")
        point_output = client.point(image, "person")
        print("Points:", point_output.get("points"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Point test failed:", e)


if __name__ == "__main__":
    image_path = "moondream/assets/how-to-be-a-people-person-1662995088.jpg"
    main(image_path)
