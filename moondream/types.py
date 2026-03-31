from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generator, List, Literal, Optional, Sequence, TypedDict, Union

from PIL import Image


@dataclass
class EncodedImage(ABC):
    pass


@dataclass
class Base64EncodedImage(EncodedImage):
    image_url: str


SamplingSettings = TypedDict(
    "SamplingSettings",
    {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "max_objects": int,
    },
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
        "points": List[List[int]],
    },
)

Reasoning = TypedDict(
    "Reasoning",
    {
        "text": str,
        "grounding": List[ReasoningGrounding],
    },
)

QueryOutput = TypedDict(
    "QueryOutput",
    {
        "answer": Union[str, Generator[str, None, None]],
        "reasoning": Optional[Reasoning],
    },
    total=False,
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

PointGroundTruth = TypedDict(
    "PointGroundTruth",
    {
        "points": List[Point],
        "boxes": List[Region],
    },
    total=False,
)

DetectGroundTruth = TypedDict("DetectGroundTruth", {"boxes": List[Region]})

FinetuneGroundTruth = Union[PointGroundTruth, DetectGroundTruth]

QueryTarget = TypedDict(
    "QueryTarget",
    {
        "answer": str,
        "reasoning": Reasoning,
    },
    total=False,
)

PointTarget = TypedDict(
    "PointTarget",
    {
        "points": List[Point],
        "boxes": List[Region],
    },
    total=False,
)

DetectTarget = TypedDict("DetectTarget", {"boxes": List[Region]})

SFTTarget = Union[QueryTarget, PointTarget, DetectTarget]

_RawRolloutOutput = TypedDict(
    "_RawRolloutOutput",
    {
        "answer": str,
        "reasoning": Optional[Reasoning],
        "points": List[Point],
        "objects": List[Region],
    },
    total=False,
)

_RawRollout = TypedDict(
    "_RawRollout",
    {
        "skill": Literal["query", "point", "detect"],
        "finish_reason": str,
        "output": _RawRolloutOutput,
        "answer_tokens": List[int],
        "thinking_tokens": List[int],
        "has_answer_separator": bool,
        "coords": List[object],
        "sizes": List[object],
    },
    total=False,
)

RolloutOutput = Union[QueryOutput, PointOutput, DetectOutput]

TrainStepOutput = TypedDict(
    "TrainStepOutput",
    {
        "step": int,
        "applied": bool,
        "kl": Optional[float],
        "router_kl": Optional[float],
        "grad_norm": Optional[float],
        "sft_loss": Optional[float],
        "reward_mean": Optional[float],
        "reward_std": Optional[float],
    },
    total=False,
)

FinetuneInfo = TypedDict(
    "FinetuneInfo",
    {
        "finetune_id": str,
        "name": str,
        "rank": int,
        "created_at_ms": int,
        "updated_at_ms": int,
    },
    total=False,
)

CheckpointInfo = TypedDict(
    "CheckpointInfo",
    {
        "checkpoint_id": str,
        "finetune_id": str,
        "step": int,
        "expires_at_ms": Optional[int],
        "created_at_ms": int,
        "updated_at_ms": int,
    },
    total=False,
)

CheckpointListOutput = TypedDict(
    "CheckpointListOutput",
    {
        "checkpoints": List[CheckpointInfo],
        "next_cursor": Optional[str],
        "has_more": bool,
    },
    total=False,
)

CheckpointDownload = TypedDict(
    "CheckpointDownload",
    {
        "url": str,
        "expires_in": int,
    },
)


@dataclass(frozen=True)
class RolloutGroup:
    skill: Literal["query", "point", "detect"]
    num_rollouts: int = 1
    image: Optional[Union[Image.Image, EncodedImage]] = None
    question: Optional[str] = None
    object: Optional[str] = None
    spatial_refs: Optional[List[SpatialRef]] = None
    reasoning: bool = False
    settings: Optional[SamplingSettings] = None
    ground_truth: Optional[FinetuneGroundTruth] = None

    @classmethod
    def query(
        cls,
        question: str,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        *,
        num_rollouts: int = 1,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        spatial_refs: Optional[List[SpatialRef]] = None,
    ) -> "RolloutGroup":
        return cls(
            skill="query",
            num_rollouts=num_rollouts,
            image=image,
            question=question,
            spatial_refs=spatial_refs,
            reasoning=reasoning,
            settings=settings,
        )

    @classmethod
    def point(
        cls,
        image: Union[Image.Image, EncodedImage],
        object: str,
        *,
        num_rollouts: int = 1,
        settings: Optional[SamplingSettings] = None,
        ground_truth: Optional[PointGroundTruth] = None,
    ) -> "RolloutGroup":
        return cls(
            skill="point",
            num_rollouts=num_rollouts,
            image=image,
            object=object,
            settings=settings,
            ground_truth=ground_truth,
        )

    @classmethod
    def detect(
        cls,
        image: Union[Image.Image, EncodedImage],
        object: str,
        *,
        num_rollouts: int = 1,
        settings: Optional[SamplingSettings] = None,
        ground_truth: Optional[DetectGroundTruth] = None,
    ) -> "RolloutGroup":
        return cls(
            skill="detect",
            num_rollouts=num_rollouts,
            image=image,
            object=object,
            settings=settings,
            ground_truth=ground_truth,
        )


@dataclass(frozen=True)
class RLGroup:
    skill: Literal["query", "point", "detect"]
    rollouts: List[RolloutOutput]
    image: Optional[Union[Image.Image, EncodedImage]] = None
    question: Optional[str] = None
    object: Optional[str] = None
    spatial_refs: Optional[List[SpatialRef]] = None
    reasoning: bool = False
    settings: Optional[SamplingSettings] = None
    rewards: Optional[List[float]] = None
    _request_payload: Optional[dict] = field(default=None, repr=False, compare=False)
    _rollouts_payload: Optional[List[_RawRollout]] = field(
        default=None, repr=False, compare=False
    )

    def with_rewards(self, rewards: Sequence[float]) -> "RLGroup":
        rewards_list = list(rewards)
        if len(rewards_list) != len(self.rollouts):
            raise ValueError("rewards must match rollouts length")
        return replace(self, rewards=rewards_list)


@dataclass(frozen=True)
class SFTGroup:
    skill: Literal["query", "point", "detect"]
    targets: List[SFTTarget]
    image: Optional[Union[Image.Image, EncodedImage]] = None
    question: Optional[str] = None
    object: Optional[str] = None
    spatial_refs: Optional[List[SpatialRef]] = None
    reasoning: bool = False
    settings: Optional[SamplingSettings] = None

    @classmethod
    def query(
        cls,
        question: str,
        targets: Sequence[QueryTarget],
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        *,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        spatial_refs: Optional[List[SpatialRef]] = None,
    ) -> "SFTGroup":
        return cls(
            skill="query",
            targets=list(targets),
            image=image,
            question=question,
            spatial_refs=spatial_refs,
            reasoning=reasoning,
            settings=settings,
        )

    @classmethod
    def point(
        cls,
        image: Union[Image.Image, EncodedImage],
        object: str,
        targets: Sequence[PointTarget],
        *,
        settings: Optional[SamplingSettings] = None,
    ) -> "SFTGroup":
        return cls(
            skill="point",
            targets=list(targets),
            image=image,
            object=object,
            settings=settings,
        )

    @classmethod
    def detect(
        cls,
        image: Union[Image.Image, EncodedImage],
        object: str,
        targets: Sequence[DetectTarget],
        *,
        settings: Optional[SamplingSettings] = None,
    ) -> "SFTGroup":
        return cls(
            skill="detect",
            targets=list(targets),
            image=image,
            object=object,
            settings=settings,
        )


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
