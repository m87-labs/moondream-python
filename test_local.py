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
    client = md.vl(api_url="http://localhost:8000")

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
        for chunk in client.caption(image, stream=True)["caption"]:
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
    image_path = "/workspace/point_max300_md10_s1201.jpg"
    main(image_path)
