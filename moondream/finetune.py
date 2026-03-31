import base64
import json
import queue
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from importlib.metadata import version as _pkg_version
from typing import Dict, Generator, Iterable, List, Mapping, Optional, Sequence, Union

from PIL import Image

from .types import (
    Base64EncodedImage,
    CheckpointListOutput,
    FinetuneGroundTruth,
    FinetuneInfo,
    EncodedImage,
    MetricsLogOutput,
    RLGroup,
    RolloutsResponse,
    SamplingSettings,
    SaveCheckpointOutput,
    SFTTarget,
    Skill,
    SkillRequest,
    SFTGroup,
    SpatialRef,
    TrainStepOutput,
)

__version__ = _pkg_version("moondream")

DEFAULT_TUNING_ENDPOINT = "https://api.moondream.ai/v1/tuning"

_RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def _encode_image(image) -> Base64EncodedImage:
    if isinstance(image, Base64EncodedImage):
        return image
    if not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return Base64EncodedImage(image_url=f"data:image/jpeg;base64,{img_str}")
    except Exception as exc:
        raise ValueError("Failed to convert image to JPEG.") from exc


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRY_STATUS_CODES
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (TimeoutError, socket.timeout))


_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 8.0
_REQUEST_TIMEOUT = 60.0


class Finetune:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        endpoint: str,
        finetune_id: str,
        name: str,
        rank: int,
    ):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.finetune_id = finetune_id
        self.name = name
        self.rank = rank

    def _headers(self, has_body: bool = False) -> Dict[str, str]:
        headers = {
            "User-Agent": f"moondream-python/{__version__}",
        }
        if has_body:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-Moondream-Auth"] = self.api_key
        return headers

    def _url(self, path: str, query: Optional[dict] = None) -> str:
        url = f"{self.endpoint}{path}"
        if query:
            items = [(k, v) for k, v in query.items() if v is not None]
            if items:
                url = f"{url}?{urllib.parse.urlencode(items)}"
        return url

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    self._url(path, query=query),
                    data=data,
                    headers=self._headers(has_body=payload is not None),
                    method=method,
                )
                with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as response:
                    body = response.read()
                    if not body:
                        return {}
                    return json.loads(body.decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                    raise
                max_delay = min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * (2 ** attempt))
                time.sleep(random.uniform(0.0, max_delay))
        raise last_exc  # unreachable, but keeps the type checker happy

    def _request_payload(
        self,
        *,
        skill: Skill,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        object: Optional[str] = None,
        spatial_refs: Optional[Sequence[SpatialRef]] = None,
        reasoning: bool = False,
        settings: Optional[Mapping[str, object]] = None,
    ) -> SkillRequest:
        payload: SkillRequest = {"skill": skill}
        if image is not None:
            payload["image_url"] = _encode_image(image).image_url
        if question is not None:
            payload["question"] = question
        if object is not None:
            payload["object"] = object
        if spatial_refs is not None:
            payload["spatial_refs"] = list(spatial_refs)
        if reasoning:
            payload["reasoning"] = True
        if settings is not None:
            payload["settings"] = dict(settings)
        return payload

    def rollouts(
        self,
        skill: Skill,
        *,
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        object: Optional[str] = None,
        num_rollouts: int = 1,
        settings: Optional[SamplingSettings] = None,
        reasoning: bool = False,
        spatial_refs: Optional[Sequence[SpatialRef]] = None,
        ground_truth: Optional[FinetuneGroundTruth] = None,
    ) -> RolloutsResponse:
        """Generate rollouts for a single request.

        Returns the raw `/rollouts` response with `request`, `rollouts`, and
        optional `rewards`.
        """
        payload: dict = {
            "finetune_id": self.finetune_id,
            "num_rollouts": num_rollouts,
            "request": self._request_payload(
                skill=skill,
                image=image,
                question=question,
                object=object,
                spatial_refs=spatial_refs,
                reasoning=reasoning,
                settings=settings,
            ),
        }
        if ground_truth is not None:
            payload["ground_truth"] = dict(ground_truth)
        return self._request_json("POST", "/rollouts", payload=payload)

    def delete(self) -> None:
        self._request_json("DELETE", f"/finetunes/{self.finetune_id}")

    def rollout_stream(
        self,
        requests: Iterable[tuple],
        *,
        max_concurrency: int = 4,
        buffer_size: int = 8,
    ) -> Generator[tuple, None, None]:
        """Generate rollouts in the background, yielding results as they complete.

        Takes an iterable of ``(context, RolloutRequest)`` tuples and yields
        ``(context, RolloutsResponse)`` tuples.  The context is passed through
        untouched so callers can pair responses with ground-truth labels or
        other metadata needed for scoring.

        Rollout requests are dispatched from background threads so that
        the caller can run train_step while the next batch of rollouts is
        already in flight.  The bounded buffer provides backpressure so
        generation never gets too far ahead of training.

        Results are yielded in completion order, not submission order.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")

        result_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        requests_iter = iter(requests)
        requests_lock = threading.Lock()
        stop = threading.Event()
        remaining = [max_concurrency]
        remaining_lock = threading.Lock()
        _DONE = object()

        def worker():
            try:
                while True:
                    with requests_lock:
                        if stop.is_set():
                            return
                        try:
                            context, request = next(requests_iter)
                        except StopIteration:
                            return
                        except Exception as exc:
                            stop.set()
                            while True:
                                try:
                                    result_queue.put(exc, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                            return

                    try:
                        response = self.rollouts(**request)
                    except Exception as exc:
                        stop.set()
                        while True:
                            try:
                                result_queue.put(exc, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                        return

                    while not stop.is_set():
                        try:
                            result_queue.put((context, response), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            finally:
                with remaining_lock:
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        result_queue.put(_DONE)

        threads = []
        for _ in range(max_concurrency):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        try:
            while True:
                item = result_queue.get()
                if item is _DONE:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            while True:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
            for t in threads:
                t.join(timeout=5.0)

    def build_sft_group(
        self,
        *,
        skill: Skill,
        targets: Sequence[SFTTarget],
        image: Optional[Union[Image.Image, EncodedImage]] = None,
        question: Optional[str] = None,
        object: Optional[str] = None,
        reasoning: bool = False,
        spatial_refs: Optional[Sequence[SpatialRef]] = None,
    ) -> SFTGroup:
        return {
            "mode": "sft",
            "request": self._request_payload(
                skill=skill,
                image=image,
                question=question,
                object=object,
                spatial_refs=spatial_refs,
                reasoning=reasoning,
            ),
            "targets": list(targets),
        }

    def train_step(
        self,
        groups: Sequence[Union[RLGroup, SFTGroup]],
        lr: float = 0.002,
    ) -> TrainStepOutput:
        payload = {
            "finetune_id": self.finetune_id,
            "groups": list(groups),
            "lr": lr,
        }
        result = self._request_json("POST", "/train_step", payload=payload)
        return result

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, float],
    ) -> MetricsLogOutput:
        """Log user-defined metrics for a training step.

        Metric names must match `[A-Za-z0-9_/-]+`, cannot start with `sys/` or
        `usr/`, and values must be finite numbers.

        Example:
            ft.log_metrics(
                step=step_output["step"],
                metrics={
                    "eval/country_match": 0.63,
                    "eval/token_f1": 0.64,
                },
            )
        """
        return self._request_json(
            "POST",
            f"/finetunes/{self.finetune_id}/metrics",
            payload={"step": step, "metrics": dict(metrics)},
        )

    def list_checkpoints(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> CheckpointListOutput:
        return self._request_json(
            "GET",
            f"/finetunes/{self.finetune_id}/checkpoints",
            query={"limit": limit, "cursor": cursor},
        )

    def save_checkpoint(self) -> SaveCheckpointOutput:
        return self._request_json(
            "POST", f"/finetunes/{self.finetune_id}/checkpoints/save"
        )

    def delete_checkpoint(self, step: int) -> None:
        self._request_json(
            "DELETE", f"/finetunes/{self.finetune_id}/checkpoints/{step}"
        )

    def model(self, step: int) -> str:
        return f"moondream3-preview/{self.finetune_id}@{step}"


def ft(
    api_key: Optional[str] = None,
    *,
    name: Optional[str] = None,
    rank: Optional[int] = None,
    finetune_id: Optional[str] = None,
    endpoint: str = DEFAULT_TUNING_ENDPOINT,
) -> Finetune:
    if finetune_id is not None:
        if name is not None or rank is not None:
            raise ValueError("finetune_id cannot be combined with name or rank")
        client = Finetune(
            api_key=api_key,
            endpoint=endpoint,
            finetune_id=finetune_id,
            name="",
            rank=0,
        )
        result = client._request_json("GET", f"/finetunes/{finetune_id}")
        finetune: FinetuneInfo = result.get("finetune", result)
        client.finetune_id = finetune["finetune_id"]
        client.name = finetune["name"]
        client.rank = finetune["rank"]
        return client

    if name is None or rank is None:
        raise ValueError("ft requires either finetune_id or both name and rank")

    client = Finetune(
        api_key=api_key,
        endpoint=endpoint,
        finetune_id="",
        name=name,
        rank=rank,
    )
    result = client._request_json(
        "POST",
        "/finetunes",
        payload={"name": name, "rank": rank},
    )
    client.finetune_id = result["finetune_id"]
    client.name = name
    client.rank = rank
    return client
