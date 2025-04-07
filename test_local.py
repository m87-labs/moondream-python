#!/usr/bin/env python
"""
test_local_client.py

This script tests the moondream Python client in local mode.
It loads an image from a file path provided as a command-line argument,
and then calls the caption, query, detect, and point methods, printing their outputs.

Usage:
    python test_local_client.py /path/to/your/image.jpg
"""

import sys
import os
import moondream as md
from PIL import Image


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
    client = md.vl(local=True)

    # Test the caption method.
    try:
        print("starting caption")
        caption_output = client.caption(image, settings={"max_tokens": 10})
        print("Caption:", caption_output.get("caption"))
        print(caption_output)
        print("------ done ------")
    except Exception as e:
        print("Caption test failed:", e)

    try:
        print("starting caption stream")
        for chunk in client.caption(image, stream=True)["caption"]:
            print(chunk, end="", flush=True)
        print("------ done ------")
    except Exception as e:
        print("Caption stream test failed:", e)

    try:
        query_output = client.query(
            image, "What's in the image?", settings={"max_tokens": 10}
        )
        print("Query Answer:", query_output.get("answer"))
    except Exception as e:
        print("Query test failed:", e)

    try:
        print("starting query stream")
        for chunk in client.query(image, "What's in the image?", stream=True)["answer"]:
            print(chunk, end="", flush=True)
        print("------ done ------")
    except Exception as e:
        print("query stream test failed:", e)

    # Test the detect method.
    try:
        detect_output = client.detect(image, "face")
        print("Detected Objects:", detect_output.get("objects"))
    except Exception as e:
        print("Detect test failed:", e)

    # Test the point method.
    try:
        point_output = client.point(image, "person")
        print("Points:", point_output.get("points"))
    except Exception as e:
        print("Point test failed:", e)


if __name__ == "__main__":
    image_path = "/Users/ethanreid/Downloads/how-to-be-a-people-person-1662995088.jpg"
    main(image_path)
