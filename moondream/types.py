from abc import ABC, abstractmethod
from PIL import Image
from dataclasses import dataclass
from typing import Generator, List, TypedDict, Union, Optional, Literal


@dataclass
class EncodedImage(ABC):
    pass


@dataclass
class Base64EncodedImage(EncodedImage):
    image_url: str


SamplingSettings = TypedDict(
    "SamplingSettings",
    {"max_tokens": int},
    total=False,
)

CaptionOutput = TypedDict(
    "CaptionOutput", {"caption": Union[str, Generator[str, None, None]]}
)

ReasoningGrounding = TypedDict(
    "ReasoningGrounding",
    {
        "start_idx": int,
        "end_idx": int,
        "points": List[List[int]]  # List of [x, y] coordinate pairs
    }
)

Reasoning = TypedDict(
    "Reasoning",
    {
        "text": str,
        "grounding": List[ReasoningGrounding]
    }
)

QueryOutput = TypedDict(
    "QueryOutput", 
    {
        "answer": Union[str, Generator[str, None, None]], 
        "reasoning": Optional[Reasoning]
    },
    total=False
)

Region = TypedDict(
    "Region", {"x_min": float, "y_min": float, "x_max": int, "y_max": float}
)
DetectOutput = TypedDict("DetectOutput", {"objects": List[Region]})

Point = TypedDict("Point", {"x": float, "y": float})
PointOutput = TypedDict("PointOutput", {"points": List[Point]})

SpatialRef = List[float]  # [x, y] point or [x1, y1, x2, y2] bbox, normalized to [0, 1]

SegmentOutput = TypedDict(
    "SegmentOutput",
    {
        "path": str,
        "bbox": Optional[Region],
    },
    total=False,
)

# Streaming segment yields these update dicts
SegmentStreamChunk = TypedDict(
    "SegmentStreamChunk",
    {
        "bbox": Optional[Region],  # Present in first message and final message
        "chunk": Optional[str],  # Coarse path chunk (path_delta messages)
        "path": Optional[str],  # Final refined path (final message only)
        "completed": Optional[bool],  # True in final message
    },
    total=False,
)

SegmentStreamOutput = Generator[SegmentStreamChunk, None, None]


class VLM(ABC):
    @abstractmethod
    def encode_image(self, image: Union[Image.Image, EncodedImage]) -> EncodedImage:
        """
        Preprocess the image by running it through the model. Only supported for local
        inference.

        This method is useful if the user wants to make multiple queries with the same image.
        The output is not guaranteed to be backward-compatible across version updates,
        and should not be persisted out of band.

        Args:
            image (Image.Image): The input image to be encoded.

        Returns:
            The encoded representation of the image.
        """

    @abstractmethod
    def caption(
        self,
        image: Union[Image.Image, EncodedImage],
        length: Literal["normal", "short"] = "normal",
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        variant: Optional[str] = None,
    ) -> CaptionOutput:
        """
        Generate a caption for the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be captioned.
            length (str): Length of caption to generate. Can be "normal" or "short".
                Defaults to "normal".
            stream (bool): If True, returns a generator that streams the output tokens.
                Defaults to False.
            settings (Optional[SamplingSettings]): Optional settings for the caption
                generation. If not provided, default settings will be used.

        Returns:
            CaptionOutput: A dictionary containing the 'caption' field with either a string
                or generator that yields strings for the caption.
        """

    @abstractmethod
    def query(
        self,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        variant: Optional[str] = None,
    ) -> QueryOutput:
        """
        Generate an answer to the input question about the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be queried.
            question (str): The question to be answered.
            stream (bool): If True, returns a generator that streams the output tokens.
                (default: False)
            settings (Optional[SamplingSettings]): Optional settings for the query
                generation.

        Returns:
            QueryOutput: A dictionary containing the 'answer' field with either a string
                or generator that yields strings for the response.
        """

    @abstractmethod
    def detect(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        variant: Optional[str] = None,
    ) -> DetectOutput:
        """
        Detect and localize the specified object in the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be analyzed.
            object (str): The object to be detected in the image.

        Returns:
            DetectOutput: A dictionary containing:
                'objects' (List[Region]): List of detected object regions, where each
                    Region has:
                    - x_min (float): Left boundary of detection box
                    - y_min (float): Top boundary of detection box
                    - x_max (float): Right boundary of detection box
                    - y_max (float): Bottom boundary of detection box
        """

    @abstractmethod
    def point(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        variant: Optional[str] = None,
    ) -> PointOutput:
        """
        Points out all instances of the given object in the input image.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to be analyzed for
                pointing out objects.
            object (str): The object type to be pointed out in the image.

        Returns:
            PointOutput: A dictionary containing:
                'points' (List[Point]): List of detected points, where each Point has:
                    - x (float): X coordinate of the point marking the object
                    - y (float): Y coordinate of the point marking the object

        This method identifies instances of the specified object in the image and returns
        a list of coordinates marking the location of each instance found. Each point
        indicates the approximate center or most relevant position for that object
        instance.
        """

    @abstractmethod
    def segment(
        self,
        image: Union[Image.Image, EncodedImage],
        object: str,
        spatial_refs: Optional[List[SpatialRef]] = None,
        stream: bool = False,
        settings: Optional[SamplingSettings] = None,
    ) -> Union[SegmentOutput, SegmentStreamOutput]:
        """
        Segment an object from the image and return an SVG path.

        Args:
            image (Union[Image.Image, EncodedImage]): The input image to segment.
            object (str): The object to segment from the image.
            spatial_refs (Optional[List[SpatialRef]]): Optional spatial references to guide
                segmentation. Each ref is either a [x, y] point or [x1, y1, x2, y2] bbox,
                with values normalized to [0, 1].
            stream (bool): If True, returns a generator yielding update dicts. Defaults to False.
            settings (Optional[SamplingSettings]): Optional settings for the segmentation.

        Returns:
            When stream=False: SegmentOutput dict with 'path' and 'bbox'.
            When stream=True: Generator yielding SegmentStreamChunk dicts:
                - {"bbox": Region} - bounding box (first message)
                - {"chunk": str} - coarse path chunks
                - {"path": str, "bbox": Region, "completed": True} - final refined path
        """
