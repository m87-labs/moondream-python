#!/usr/bin/env python
"""
Test Both Clients and Compare Results

This script tests the moondream Python client by sending the same prompt through both the cloud and local endpoints.
It loads an image from a file path provided as a command-line argument, then calls the caption, query, detect,
and point methods for each endpoint. It prints the outputs and checks whether the responses are identical.

Usage:
    python test_both_clients.py /path/to/your/image.jpg

Note:
    Replace 'YOUR_API_KEY' with your actual API key for the cloud endpoint.
"""

import sys
import os
import moondream as md
from PIL import Image


def compare_outputs(name: str, cloud_val, local_val):
    print(f"Cloud {name}: {cloud_val}")
    print(f"Local {name}: {local_val}")
    if cloud_val == local_val:
        print(f"Result: The {name} outputs are identical.\n")
    else:
        print(f"Result: The {name} outputs differ.\n")


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

    # Instantiate the clients
    # Cloud client uses an API key; ensure you replace 'YOUR_API_KEY' with a valid key
    cloud_client = md.vl(
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI3MDI4ODY0Yy01ZmJkLTQ4ZDItYmUyMC0wNjcxOTQwNmRiMTQiLCJpYXQiOjE3NDM5NzkwODF9.1HoCbUW9mGYbrSdM_0iKo8DxXe7xVF1dplKKtlRoasU"
    )
    # Local client uses the local endpoint
    local_client = md.vl(local=True)

    print("--- Testing Caption ---")
    try:
        cloud_caption = cloud_client.caption(image)
        local_caption = local_client.caption(image)
        compare_outputs(
            "Caption", cloud_caption.get("caption"), local_caption.get("caption")
        )
    except Exception as e:
        print(f"Caption test failed: {e}")

    print("--- Testing Query ---")
    try:
        query_prompt = "What's in the image?"
        cloud_query = cloud_client.query(image, query_prompt)
        local_query = local_client.query(image, query_prompt)
        compare_outputs(
            "Query Answer", cloud_query.get("answer"), local_query.get("answer")
        )
    except Exception as e:
        print(f"Query test failed: {e}")

    print("--- Testing Detect ---")
    try:
        detect_target = "face"
        cloud_detect = cloud_client.detect(image, detect_target)
        local_detect = local_client.detect(image, detect_target)
        compare_outputs(
            "Detected Objects", cloud_detect.get("objects"), local_detect.get("objects")
        )
    except Exception as e:
        print(f"Detect test failed: {e}")

    print("--- Testing Point ---")
    try:
        point_target = "person"
        cloud_point = cloud_client.point(image, point_target)
        local_point = local_client.point(image, point_target)
        compare_outputs("Points", cloud_point.get("points"), local_point.get("points"))
    except Exception as e:
        print(f"Point test failed: {e}")


if __name__ == "__main__":
    image_path = "/Users/ethanreid/Downloads/how-to-be-a-people-person-1662995088.jpg"
    main(image_path)
