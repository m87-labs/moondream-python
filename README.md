# Moondream Python Client Library

Official Python client library for Moondream, a tiny vision language model that can
analyze images and answer questions about them. This client library provides easy
access to Moondream's API endpoints for image analysis.

## Features

- **caption**: Generate descriptive captions for images
- **query**: Ask questions about image content
- **detect**: Find bounding boxes around objects in images
- **point**: Identify the center location of specified objects in images

## Installation

Install the package from PyPI:

```bash
pip install moondream
```

## Quick Start

### Cloud

- Get your free API key from [console.moondream.ai](https://console.moondream.ai).

```python
import moondream as md
from PIL import Image

# Initialize with API key
model = md.vl(api_key="your-api-key")

# Load an image
image = Image.open("path/to/image.jpg")

# Generate a caption
caption = model.caption(image)["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the response
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

### Local Inference

- Install Moondream server from: !ADD LINK TO DOWNLOAD!
- Run the local server:
  ```bash
  ./moondream-server
- Set the `api_url` parameter to the URL of the local server (the default is `http://localhost:8000`)

```python
import moondream as md
from PIL import Image

# Initialize with local api_url
model = md.vl(api_url="http://localhost:8000")

# Load an image
image = Image.open("path/to/image.jpg")

# Generate a caption
caption = model.caption(image)["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the response
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

## API Reference

### Constructor

```python
# Cloud inference
model = md.vl(api_key="your-api-key")

# Local inference
model = md.vl(api_url="http://localhost:8000")
```

### Methods

#### caption(self, image: Union[Image.Image, EncodedImage], length: Literal["normal", "short", "long"] = "normal", stream: bool = False) -> CaptionOutput

Generate a caption for an image.

```python
caption = model.caption(image, length="short")["caption"]
print(caption)

# Generate a caption with streaming (default: False)
for chunk in model.caption(image, length="short", stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

#### query(self, image: Union[Image.Image, EncodedImage], question: str, stream: bool = False) -> QueryOutput

Ask a question about an image.

```python
answer = model.query(image, question="What's in this image?")["answer"]
print("Answer:", answer)

# Ask a question with streaming (default: False)
for chunk in model.query(image, question="What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)
```

#### detect(self, image: Union[Image.Image, EncodedImage], object: str) -> DetectOutput

Detect specific objects in an image.

```python
detect_output = model.detect(image, "item")["objects"]
print(detect_output)
```

#### point(self, image: Union[Image.Image, EncodedImage], object: str) -> PointOutput

Get coordinates of specific objects in an image.

```python
point_output = model.point(image, "person")
print(point_output)
```

#### encode_image(self, image: Union[Image.Image, EncodedImage]) -> Base64EncodedImage

Produce Base64EncodedImage.

```python
encoded_image = model.encode_image(image)
```

### Image Types

- Image.Image: PIL Image object
- Base64EncodedImage: Object produced by model.encode_image(image), subtype of EncodedImage

### Response Types

- CaptionOutput: `{"caption": str | Generator}`
- QueryOutput: `{"answer": str | Generator}`
- DetectOutput: `{"objects": List[Region]}`
- PointOutput: `{"points": List[Point]}`
- Region: Bounding box with coordinates (`x_min`, `y_min`, `x_max`, `y_max`)
- Point: Coordinates (`x`, `y`) indicating the object center

## Links

- [Website](https://moondream.ai/)
- [Demo](https://moondream.ai/playground)
