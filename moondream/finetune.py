import asyncio
import base64
import json
import math
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from importlib.metadata import version as _pkg_version
from typing import Dict, List, Mapping, Optional, Sequence, Union

from PIL import Image

from .types import (
    Base64EncodedImage,
    CheckpointInfo,
    CheckpointListOutput,
    FinetuneInfo,
    EncodedImage,
    MetricsLogOutput,
    RLGroup,
    RolloutRequest,
    RolloutsResponse,
    SkillRequest,
    SFTGroup,
    TrainStepOutput,
)

__version__ = _pkg_version("moondream")

DEFAULT_TUNING_ENDPOINT = "https://api.moondream.ai/v1/tuning"

_METRIC_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_/-]+$")
_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504, 524}
_TRAIN_STEP_OUTPUT_KEYS = (
    "step",
    "applied",
    "kl",
    "router_kl",
    "grad_norm",
    "sft_loss",
    "reward_mean",
    "reward_std",
)


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

    async def _request_json_async(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
    ) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._request_json_once,
                    method,
                    path,
                    payload,
                    query,
                )
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt == self.max_retries:
                    if isinstance(exc, urllib.error.HTTPError):
                        _raise_http_error(path, exc)
                    raise FinetuneAPIError(f"{path} failed: {exc}") from exc
                if isinstance(exc, urllib.error.HTTPError):
                    _close_http_error(exc)
                await asyncio.sleep(self._backoff_delay(attempt))

        raise FinetuneAPIError(f"{path} failed: {last_error}")

    def _request_payload(self, request: RolloutRequest) -> SkillRequest:
        payload: SkillRequest = {"skill": request.skill}
        if request.image is not None:
            payload["image_url"] = _encode_image(request.image).image_url
        if request.question is not None:
            payload["question"] = request.question
        if request.object is not None:
            payload["object"] = request.object
        if request.spatial_refs is not None:
            payload["spatial_refs"] = list(request.spatial_refs)
        if request.reasoning:
            payload["reasoning"] = True
        if request.settings is not None:
            payload["settings"] = dict(request.settings)
        return payload

    def _rollouts_payload(self, request: RolloutRequest) -> dict:
        payload = {
            "finetune_id": self.finetune_id,
            "num_rollouts": request.num_rollouts,
            "request": self._request_payload(request),
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

    async def _rollouts_async(self, request: RolloutRequest) -> RolloutsResponse:
        return await self._request_json_async(
            "POST",
            "/rollouts",
            payload=self._rollouts_payload(request),
        )

    def batch_rollouts(
        self,
        requests: Sequence[RolloutRequest],
        max_concurrency: int = 4,
    ) -> List[RLGroup]:
        """Generate multiple rollout requests in parallel.

        Returns RL groups with `mode`, `request`, and `rollouts` populated.
        Fill `group["rewards"]` before calling `train_step(...)`.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        requests_list = list(requests)
        if not requests_list:
            return []
        return self._run_async(
            self._batch_rollouts_async(requests_list, max_concurrency=max_concurrency)
        )

    async def _batch_rollouts_async(
        self,
        requests: Sequence[RolloutRequest],
        max_concurrency: int,
    ) -> List[RLGroup]:
        results: List[Optional[RLGroup]] = [None] * len(requests)
        next_index = 0
        first_error: Optional[BaseException] = None
        lock = asyncio.Lock()
        stop = asyncio.Event()

        async def worker():
            nonlocal next_index, first_error

            while True:
                async with lock:
                    if stop.is_set() or next_index >= len(requests):
                        return
                    index = next_index
                    next_index += 1

                try:
                    response = await self._rollouts_async(requests[index])
                    results[index] = self._rl_group_from_response(response)
                except Exception as exc:
                    if not stop.is_set():
                        first_error = exc
                        stop.set()
                    return

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(max_concurrency, len(requests)))
        ]
        await asyncio.gather(*tasks)

        if first_error is not None:
            raise first_error

        return [result for result in results if result is not None]

    def train_step(
        self,
        groups: Sequence[Union[RLGroup, SFTGroup]],
        lr: float = 0.002,
    ) -> TrainStepOutput:
        if not groups:
            raise ValueError("train_step requires at least one group")

        payload_groups = []
        for group in groups:
            mode = group.get("mode")
            if mode == "rl":
                rewards = group.get("rewards")
                rollouts = group.get("rollouts")
                if rewards is None:
                    raise ValueError("RLGroup rewards must be set before train_step")
                if rollouts is None:
                    raise ValueError("RLGroup requires rollouts")
                if len(rewards) != len(rollouts):
                    raise ValueError("rewards must match rollouts length")
                payload_groups.append(group)
                continue

            if mode == "sft":
                payload_groups.append(group)
                continue

            raise ValueError("train_step groups must have mode 'rl' or 'sft'")

        payload = {
            "finetune_id": self.finetune_id,
            "groups": payload_groups,
            "lr": lr,
        }
        result = self._request_json(
            "POST",
            "/train_step",
            payload=payload,
            max_retries=0,
        )
        return {key: result[key] for key in _TRAIN_STEP_OUTPUT_KEYS if key in result}

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
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")

        metrics_dict = dict(metrics)
        if not metrics_dict:
            raise ValueError("metrics must include at least one entry")
        if len(metrics_dict) > 100:
            raise ValueError("metrics cannot include more than 100 entries")

        payload_metrics = {}
        for name, value in metrics_dict.items():
            if not _METRIC_NAME_PATTERN.fullmatch(name):
                raise ValueError(
                    "metric names must use only letters, numbers, underscores, slashes, or hyphens"
                )
            if name.startswith("sys/") or name.startswith("usr/"):
                raise ValueError("metric names cannot start with sys/ or usr/")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("metric values must be finite numbers")
            metric_value = float(value)
            if not math.isfinite(metric_value):
                raise ValueError("metric values must be finite numbers")
            payload_metrics[name] = metric_value

        return self._request_json(
            "POST",
            f"/finetunes/{self.finetune_id}/metrics",
            payload={"step": step, "metrics": payload_metrics},
        )

    def list_checkpoints(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> CheckpointListOutput:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        return self._request_json(
            "GET",
            f"/finetunes/{self.finetune_id}/checkpoints",
            query={"limit": limit, "cursor": cursor},
        )

    def save_checkpoint(self) -> CheckpointInfo:
        result = self._request_json(
            "POST", f"/finetunes/{self.finetune_id}/checkpoints/save"
        )
        return result["checkpoint"]

    def delete_checkpoint(self, step: int) -> None:
        self._request_json(
            "DELETE", f"/finetunes/{self.finetune_id}/checkpoints/{step}"
        )

    def model(self, step: int) -> str:
        if step < 0:
            raise ValueError("step must be non-negative")
        return f"moondream3-preview/{self.finetune_id}@{step}"

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result["value"]


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
