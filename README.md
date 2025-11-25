# Moondream Python Client Library

Official Python client library for Moondream, a fast multi-function VLM. This client can target either [Moondream Cloud](https://moondream.ai/cloud) or [Moondream Station](https://moondream.ai/station).

## Capabilities

Moondream goes beyond the typical VLM "query" ability to include more visual functions:

| Method | Description |
|--------|-------------|
| `caption` | Generate descriptive captions for images |
| `query` | Ask questions about image content |
| `detect` | Find bounding boxes around objects in images |
| `point` | Identify the center location of specified objects |
| `segment` | Generate an SVG path segmentation mask for objects |

Try it out on [Moondream's playground](https://moondream.ai/playground).

## Installation

```bash
pip install moondream
```

## Quick Start

Choose how you want to run Moondream:

1. **Moondream Cloud** — Get an API key from the [cloud console](https://moondream.ai/c/cloud/api-keys)
2. **Moondream Station** — Run locally by installing [Moondream Station](https://moondream.ai/station)

```python
import moondream as md
from PIL import Image

# Initialize with Moondream Cloud
model = md.vl(api_key="<your-api-key>")

# Or initialize with a local Moondream Station
model = md.vl(endpoint="http://localhost:2020/v1")

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
model = md.vl(api_key="<your-api-key>")             # Cloud
model = md.vl(endpoint="http://localhost:2020/v1")  # Local
```

### Methods

#### `caption(image, length="normal", stream=False)`

Generate a caption for an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `length` — `"normal"`, `"short"`, or `"long"` (default: `"normal"`)
- `stream` — `bool` (default: `False`)

**Returns:** `CaptionOutput` — `{"caption": str | Generator}`

```python
caption = model.caption(image, length="short")["caption"]

# With streaming
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

---

#### `query(image, question, stream=False)`

Ask a question about an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `question` — `str`
- `stream` — `bool` (default: `False`)

**Returns:** `QueryOutput` — `{"answer": str | Generator}`

```python
answer = model.query(image, "What's in this image?")["answer"]

# With streaming
for chunk in model.query(image, "What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)
```

---

#### `detect(image, object)`

Detect specific objects in an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `object` — `str`

**Returns:** `DetectOutput` — `{"objects": List[Region]}`

```python
objects = model.detect(image, "car")["objects"]
```

---

#### `point(image, object)`

Get coordinates of specific objects in an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `object` — `str`

**Returns:** `PointOutput` — `{"points": List[Point]}`

```python
points = model.point(image, "person")["points"]
```

---

#### `segment(image, object, spatial_refs=None, stream=False)`

Segment an object from an image and return an SVG path.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `object` — `str`
- `spatial_refs` — `List[[x, y] | [x1, y1, x2, y2]]` — optional spatial hints (normalized 0-1)
- `stream` — `bool` (default: `False`)

**Returns:**
- Non-streaming: `SegmentOutput` — `{"path": str, "bbox": Region}`
- Streaming: Generator yielding update dicts

```python
result = model.segment(image, "cat")
svg_path = result["path"]
bbox = result["bbox"]  # {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}

# With spatial hint (point)
result = model.segment(image, "cat", spatial_refs=[[0.5, 0.5]])

# With streaming
for update in model.segment(image, "cat", stream=True):
    if "bbox" in update and not update.get("completed"):
        print(f"Bbox: {update['bbox']}")  # Available in first message
    if "chunk" in update:
        print(update["chunk"], end="")  # Coarse path chunks
    if update.get("completed"):
        print(f"Final path: {update['path']}")  # Refined path
        print(f"Final bbox: {update['bbox']}")
```

---

#### `encode_image(image)`

Pre-encode an image for reuse across multiple calls.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`

**Returns:** `Base64EncodedImage`

```python
encoded = model.encode_image(image)
```

### Types

| Type | Description |
|------|-------------|
| `Image.Image` | PIL Image object |
| `EncodedImage` | Base class for encoded images |
| `Base64EncodedImage` | Output of `encode_image()`, subtype of `EncodedImage` |
| `Region` | Bounding box with `x_min`, `y_min`, `x_max`, `y_max` |
| `Point` | Coordinates with `x`, `y` indicating object center |
| `SpatialRef` | `[x, y]` point or `[x1, y1, x2, y2]` bbox, normalized to [0, 1] |

## Links

- [Website](https://moondream.ai/)
- [Playground](https://moondream.ai/playground)
- [GitHub](https://github.com/vikhyat/moondream)
