import sys
import os
import json

import moondream as md
from PIL import Image


def test_cloud_chat_preserves_service_reasoning_default(monkeypatch):
    payloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request):
        payloads.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = md.vl(api_key="key")

    result = client.chat([{"role": "user", "content": "question"}])
    client.chat(
        [{"role": "user", "content": "question"}],
        reasoning=False,
    )

    assert "reasoning" not in payloads[0]
    assert payloads[1]["reasoning"] is False
    assert result["finish_reason"] == "stop"


def main(image_path: str):
    # Verify the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Instantiate the client in local mode.
    client = md.vl(os.environ["MOONDREAM_API_KEY"])

    # Test the caption method.
    try:
        print("starting caption")
        caption_output = client.caption(image)
        print("Caption:", caption_output.get("caption"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption test failed:", e)

    try:
        print("starting caption stream")
        for chunk in client.caption(image, length="long", stream=True)["caption"]:
            print(chunk, end="", flush=True)
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption test failed:", e)

    # Test the query method.
    try:
        print("starting query")
        query_output = client.query(image, "What's in the image?")
        print("Query Answer:", query_output.get("answer"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Query test failed:", e)

    # Test the detect method.
    try:
        print("starting detect")
        detect_output = client.detect(image, "face")
        print("Detected Objects:", detect_output.get("objects"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Detect test failed:", e)

    # Test the point method.
    try:
        print("starting point")
        point_output = client.point(image, "person")
        print("Points:", point_output.get("points"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Point test failed:", e)


if __name__ == "__main__":
    image_path = "moondream/assets/how-to-be-a-people-person-1662995088.jpg"
    main(image_path)
