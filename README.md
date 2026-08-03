# Moondream Python Client Library

Official Python client library for Moondream, a fast multi-function VLM. This client can target [Moondream Cloud](https://moondream.ai/cloud) or run locally via Photon — on NVIDIA GPUs (Linux x86_64 / aarch64 or Windows) or Apple Silicon Macs.

## Capabilities

Moondream goes beyond the typical VLM "query" ability to include more visual functions:

| Method | Description |
|--------|-------------|
| `caption` | Generate descriptive captions for images |
| `query` | Ask questions about image content |
| `chat` | Continue multi-turn conversations with text and images |
| `detect` | Find bounding boxes around objects in images |
| `point` | Identify the center location of specified objects |
| `segment` | Generate an SVG path segmentation mask for objects |

Try it out on [Moondream's playground](https://moondream.ai/playground).

## Photon Models

Photon local inference includes all models bundled with Kestrel 0.5:

| Family | Models |
|--------|--------|
| Moondream | Moondream 2, Moondream 3, Moondream 3.1 9B A2B |
| Qwen 3.5 | 0.8B, 2B, 4B, 9B, 27B, and 35B-A3B; Base variants where published |
| Qwen 3.6 | 27B and 35B-A3B; BF16 and FP8 checkpoints |
| Gemma 4 | E2B, E4B, and 31B base/instruction variants |

Use `md.photon_models()` to inspect the exact registered identifiers in the installed
release. The returned client reports `model_id`, `tasks`, and
`supports(task)` without requiring a Kestrel import.
Existing `md.vl(local=True, model=..., ...)` calls remain supported and delegate
to `md.photon(...)`.

## Installation

```bash
pip install moondream
```

## Quick Start

Choose how you want to run Moondream:

1. **Moondream Cloud** — Get an API key from the [cloud console](https://moondream.ai/c/cloud/api-keys)
2. **Moondream Photon** — High-performance local inference engine on NVIDIA GPUs (Linux / Windows) or Apple Silicon Macs (macOS 13+). Base models run locally without an API key; an API key is only needed for finetuned models.

```python
import moondream as md
from PIL import Image

# Initialize with Moondream Cloud
model = md.vl(api_key="<your-api-key>")

# Or initialize Photon local inference (NVIDIA GPU or Apple Silicon)
model = md.photon("moondream3-preview")

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

# Multi-turn chat accepts OpenAI-style messages
chat = model.chat([
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What is my name?"},
])
print(chat["message"]["content"])
```

## API Reference

### Constructor

```python
model = md.vl(api_key="<your-api-key>")                        # Cloud
model = md.photon("moondream3-preview")                        # Photon with Moondream 3
model = md.vl(api_key="<your-api-key>", model="moondream3-preview/ft_id@step")  # Finetune
qwen = md.photon("Qwen/Qwen3.5-4B")
gemma = md.photon("google/gemma-4-E2B-it")
```

Photon clients share matching local engines. Call `model.close()` when an
application is finished with a client, or use
`with md.photon("moondream3-preview") as model:`
for deterministic GPU and worker cleanup.

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

#### `query(image, question, stream=False, spatial_refs=None)`

Ask a question about an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `question` — `str`
- `stream` — `bool` (default: `False`)
- `spatial_refs` — optional point or box hints, normalized to 0-1

**Returns:** `QueryOutput` — `{"answer": str | Generator}`

```python
answer = model.query(image, "What's in this image?")["answer"]

# With streaming
for chunk in model.query(image, "What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)
```

---

#### `chat(messages, stream=False, reasoning=None)`

Continue an OpenAI-style multi-turn conversation. Message content can be text
or a list of `text` and base64 `image_url` parts. When `reasoning` is omitted,
the selected model or Cloud service supplies its default.

```python
result = model.chat([
    {"role": "user", "content": "Remember that my favorite color is green."},
    {"role": "assistant", "content": "Got it."},
    {"role": "user", "content": "What is my favorite color?"},
])
print(result["message"]["content"])

for chunk in model.chat(
    [{"role": "user", "content": "Write a short poem about the moon."}],
    stream=True,
)["message"]:
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

#### `point(image, object, spatial_refs=None)`

Get coordinates of specific objects in an image.

**Parameters:**
- `image` — `Image.Image` or `EncodedImage`
- `object` — `str`
- `spatial_refs` — optional point or box hints, normalized to 0-1

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
