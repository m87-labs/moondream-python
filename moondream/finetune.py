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
    FinetuneInfo,
    EncodedImage,
    MetricsLogOutput,
    RLGroup,
    RolloutRequest,
    RolloutsResponse,
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

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504, 524}


class FinetuneAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        body: Optional[object] = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = body


def _image_to_jpeg_bytes(image) -> bytes:
    if isinstance(image, Image.Image):
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            return buffered.getvalue()
        except Exception as exc:
            raise ValueError("Failed to convert image to JPEG.") from exc

    raise ValueError(f"Unsupported EncodedImage type: {type(image)}")


def _encode_image(image) -> Base64EncodedImage:
    if isinstance(image, Base64EncodedImage):
        return image
    image_bytes = _image_to_jpeg_bytes(image)
    img_str = base64.b64encode(image_bytes).decode()
    return Base64EncodedImage(image_url=f"data:image/jpeg;base64,{img_str}")


def _error_message(exc: urllib.error.HTTPError) -> str:
    body = ""
    try:
        body = exc.read().decode("utf-8")
    except Exception:
        body = ""

    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("error", "message", "detail"):
                value = parsed.get(key)
                if isinstance(value, str) and value:
                    return value
        if body.strip():
            return body.strip()

    return exc.reason if isinstance(exc.reason, str) else str(exc.reason)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRY_STATUS_CODES
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (TimeoutError, socket.timeout))


def _close_http_error(exc: urllib.error.HTTPError):
    try:
        exc.close()
    except Exception:
        pass


def _raise_http_error(path: str, exc: urllib.error.HTTPError):
    try:
        message = _error_message(exc)
    finally:
        _close_http_error(exc)
    raise FinetuneAPIError(
        f"{path} failed with status {exc.code}: {message}",
        status=exc.code,
        body=message,
    ) from exc


class Finetune:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        endpoint: str,
        finetune_id: str,
        name: str,
        rank: int,
        max_retries: int = 5,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        timeout: float = 60.0,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay must be non-negative")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.finetune_id = finetune_id
        self.name = name
        self.rank = rank
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.timeout = timeout

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

    def _request_json_once(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path, query=query),
            data=data,
            headers=self._headers(has_body=payload is not None),
            method=method,
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = response.read()
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

    def _backoff_delay(self, attempt: int) -> float:
        max_delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
        return random.uniform(0.0, max_delay)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ) -> dict:
        last_error: Optional[Exception] = None
        retries = self.max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                return self._request_json_once(method, path, payload=payload, query=query)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt == retries:
                    if isinstance(exc, urllib.error.HTTPError):
                        _raise_http_error(path, exc)
                    raise FinetuneAPIError(f"{path} failed: {exc}") from exc
                if isinstance(exc, urllib.error.HTTPError):
                    _close_http_error(exc)
                time.sleep(self._backoff_delay(attempt))

        raise FinetuneAPIError(f"{path} failed: {last_error}")

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

    def _rollouts_payload(self, request: RolloutRequest) -> dict:
        payload = {
            "finetune_id": self.finetune_id,
            "num_rollouts": request.num_rollouts,
            "request": self._request_payload(
                skill=request.skill,
                image=request.image,
                question=request.question,
                object=request.object,
                spatial_refs=request.spatial_refs,
                reasoning=request.reasoning,
                settings=request.settings,
            ),
        }
        if request.ground_truth is not None:
            payload["ground_truth"] = dict(request.ground_truth)
        return payload

    def _rl_group_from_response(self, response: RolloutsResponse) -> RLGroup:
        group: RLGroup = {
            "mode": "rl",
            "request": response["request"],
            "rollouts": response.get("rollouts", []),
        }
        rewards = response.get("rewards")
        if rewards is not None:
            group["rewards"] = rewards
        return group

    def rollouts(self, request: RolloutRequest) -> RolloutsResponse:
        """Generate rollouts for a single request.

        Returns the raw `/rollouts` response with `request`, `rollouts`, and
        optional `rewards`.
        """
        return self._request_json(
            "POST",
            "/rollouts",
            payload=self._rollouts_payload(request),
        )

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
                while not stop.is_set():
                    with requests_lock:
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

                    if stop.is_set():
                        return

                    try:
                        response = self.rollouts(request)
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
        result = self._request_json(
            "POST",
            "/train_step",
            payload=payload,
            max_retries=0,
        )
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
    max_retries: int = 5,
    retry_base_delay: float = 0.5,
    retry_max_delay: float = 8.0,
    timeout: float = 60.0,
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
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            timeout=timeout,
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
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        timeout=timeout,
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
