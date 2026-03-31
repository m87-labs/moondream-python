import asyncio
import base64
import json
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
from typing import Dict, List, Optional, Sequence

from PIL import Image

from .types import (
    Base64EncodedImage,
    CheckpointDownload,
    CheckpointInfo,
    CheckpointListOutput,
    DetectFinetuneRequest,
    DetectGroundTruth,
    DetectTarget,
    FinetuneSamplingSettings,
    FinetuneGroundTruth,
    FinetuneInfo,
    FinetuneRequest,
    PointFinetuneRequest,
    PointGroundTruth,
    PointTarget,
    QueryFinetuneRequest,
    QueryTarget,
    RLGroup,
    RolloutSpec,
    SFTGroup,
    TrainStepOutput,
)

__version__ = _pkg_version("moondream")

DEFAULT_TUNING_ENDPOINT = "https://api.moondream.ai/v1/tuning"

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
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
    if isinstance(image, Base64EncodedImage):
        data = image.image_url
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

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


def _retry_after_seconds(headers) -> Optional[float]:
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


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


def _raise_http_error(path: str, exc: urllib.error.HTTPError):
    message = _error_message(exc)
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

    def _backoff_delay(self, attempt: int, exc: Exception) -> float:
        if isinstance(exc, urllib.error.HTTPError):
            retry_after = _retry_after_seconds(exc.headers)
            if retry_after is not None:
                return retry_after
        max_delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
        return random.uniform(0.0, max_delay)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        query: Optional[dict] = None,
    ) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_json_once(method, path, payload=payload, query=query)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt == self.max_retries:
                    if isinstance(exc, urllib.error.HTTPError):
                        _raise_http_error(path, exc)
                    raise FinetuneAPIError(f"{path} failed: {exc}") from exc
                time.sleep(self._backoff_delay(attempt, exc))

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
                await asyncio.sleep(self._backoff_delay(attempt, exc))

        raise FinetuneAPIError(f"{path} failed: {last_error}")

    def _validate_rollout_settings(
        self, skill: str, settings: Optional[FinetuneSamplingSettings]
    ) -> Optional[dict]:
        if settings is None:
            return None
        settings_dict = dict(settings)
        if skill == "query" and "max_objects" in settings_dict:
            raise ValueError("query settings do not accept max_objects")
        if skill in ("point", "detect") and "max_tokens" in settings_dict:
            raise ValueError(f"{skill} settings do not accept max_tokens")
        return settings_dict

    def _serialize_request(self, request: FinetuneRequest) -> dict:
        skill = request.get("skill")
        if skill == "query":
            return self._serialize_query_request(request)
        if skill == "point":
            return self._serialize_point_request(request)
        if skill == "detect":
            return self._serialize_detect_request(request)
        raise ValueError("request.skill must be one of 'query', 'point', or 'detect'")

    def _serialize_query_request(self, request: QueryFinetuneRequest) -> dict:
        question = request.get("question")
        if question is None:
            raise ValueError("query request requires question")

        payload = {
            "skill": "query",
            "question": question,
        }
        image = request.get("image")
        if image is not None:
            payload["image_url"] = _encode_image(image).image_url
        spatial_refs = request.get("spatial_refs")
        if spatial_refs is not None:
            payload["spatial_refs"] = spatial_refs
        reasoning = request.get("reasoning")
        if reasoning is not None:
            payload["reasoning"] = reasoning
        settings = self._validate_rollout_settings("query", request.get("settings"))
        if settings is not None:
            payload["settings"] = settings
        return payload

    def _serialize_point_request(self, request: PointFinetuneRequest) -> dict:
        object_name = request.get("object")
        image = request.get("image")
        if object_name is None:
            raise ValueError("point request requires object")
        if image is None:
            raise ValueError("point request requires image")

        payload = {
            "skill": "point",
            "object": object_name,
            "image_url": _encode_image(image).image_url,
        }
        settings = self._validate_rollout_settings("point", request.get("settings"))
        if settings is not None:
            payload["settings"] = settings
        return payload

    def _serialize_detect_request(self, request: DetectFinetuneRequest) -> dict:
        object_name = request.get("object")
        image = request.get("image")
        if object_name is None:
            raise ValueError("detect request requires object")
        if image is None:
            raise ValueError("detect request requires image")

        payload = {
            "skill": "detect",
            "object": object_name,
            "image_url": _encode_image(image).image_url,
        }
        settings = self._validate_rollout_settings("detect", request.get("settings"))
        if settings is not None:
            payload["settings"] = settings
        return payload

    def _serialize_ground_truth(
        self, skill: str, ground_truth: Optional[FinetuneGroundTruth]
    ) -> Optional[dict]:
        if ground_truth is None:
            return None
        if skill == "query":
            raise ValueError("query rollouts do not support ground_truth")
        if skill == "point":
            return self._serialize_point_ground_truth(ground_truth)  # type: ignore[arg-type]
        if skill == "detect":
            return self._serialize_detect_ground_truth(ground_truth)  # type: ignore[arg-type]
        raise ValueError(f"unsupported skill: {skill}")

    def _serialize_point_ground_truth(self, ground_truth: PointGroundTruth) -> dict:
        has_points = "points" in ground_truth
        has_boxes = "boxes" in ground_truth
        if has_points == has_boxes:
            raise ValueError("point ground_truth requires exactly one of points or boxes")
        if has_points:
            return {"points": ground_truth["points"]}
        return {"boxes": ground_truth["boxes"]}

    def _serialize_detect_ground_truth(self, ground_truth: DetectGroundTruth) -> dict:
        if "boxes" not in ground_truth:
            raise ValueError("detect ground_truth requires boxes")
        return {"boxes": ground_truth["boxes"]}

    def _serialize_targets(self, request_payload: dict, targets: Sequence[dict]) -> List[dict]:
        skill = request_payload["skill"]
        if not targets:
            raise ValueError("SFTGroup requires at least one target")
        if skill == "query":
            return [self._serialize_query_target(request_payload, target) for target in targets]
        if skill == "point":
            return [self._serialize_point_target(target) for target in targets]
        if skill == "detect":
            return [self._serialize_detect_target(target) for target in targets]
        raise ValueError(f"unsupported skill: {skill}")

    def _serialize_query_target(self, request_payload: dict, target: QueryTarget) -> dict:
        answer = target.get("answer")
        if answer is None:
            raise ValueError("query SFT targets require answer")
        payload = {"answer": answer}
        if request_payload.get("reasoning"):
            reasoning = target.get("reasoning")
            if reasoning is None:
                raise ValueError("query SFT targets require reasoning when request.reasoning is true")
            payload["reasoning"] = reasoning
        elif "reasoning" in target:
            raise ValueError("query SFT targets should omit reasoning when request.reasoning is false")
        return payload

    def _serialize_point_target(self, target: PointTarget) -> dict:
        has_points = "points" in target
        has_boxes = "boxes" in target
        if has_points == has_boxes:
            raise ValueError("point SFT targets require exactly one of points or boxes")
        if has_points:
            return {"points": target["points"]}
        return {"boxes": target["boxes"]}

    def _serialize_detect_target(self, target: DetectTarget) -> dict:
        if "boxes" not in target:
            raise ValueError("detect SFT targets require boxes")
        return {"boxes": target["boxes"]}

    def delete(self) -> None:
        self._request_json("DELETE", f"/finetunes/{self.finetune_id}")

    def rollouts(
        self,
        request: FinetuneRequest,
        num_rollouts: int = 1,
        ground_truth: Optional[FinetuneGroundTruth] = None,
    ) -> RLGroup:
        if num_rollouts < 1 or num_rollouts > 16:
            raise ValueError("num_rollouts must be between 1 and 16")

        request_payload = self._serialize_request(request)
        payload = {
            "finetune_id": self.finetune_id,
            "num_rollouts": num_rollouts,
            "request": request_payload,
        }
        ground_truth_payload = self._serialize_ground_truth(
            request_payload["skill"], ground_truth
        )
        if ground_truth_payload is not None:
            payload["ground_truth"] = ground_truth_payload

        result = self._request_json("POST", "/rollouts", payload=payload)
        return RLGroup(
            request=request,
            rollouts=result["rollouts"],
            rewards=result.get("rewards"),
            _request_payload=result.get("request", request_payload),
        )

    async def _rollouts_async(self, spec: RolloutSpec) -> RLGroup:
        if spec.num_rollouts < 1 or spec.num_rollouts > 16:
            raise ValueError("num_rollouts must be between 1 and 16")

        request_payload = self._serialize_request(spec.request)
        payload = {
            "finetune_id": self.finetune_id,
            "num_rollouts": spec.num_rollouts,
            "request": request_payload,
        }
        ground_truth_payload = self._serialize_ground_truth(
            request_payload["skill"], spec.ground_truth
        )
        if ground_truth_payload is not None:
            payload["ground_truth"] = ground_truth_payload

        result = await self._request_json_async("POST", "/rollouts", payload=payload)
        return RLGroup(
            request=spec.request,
            rollouts=result["rollouts"],
            rewards=result.get("rewards"),
            _request_payload=result.get("request", request_payload),
        )

    def rollout_groups(
        self,
        specs: Sequence[RolloutSpec],
        max_concurrency: int = 4,
    ) -> List[RLGroup]:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        specs_list = list(specs)
        if not specs_list:
            return []
        return self._run_async(
            self._rollout_groups_async(specs_list, max_concurrency=max_concurrency)
        )

    async def _rollout_groups_async(
        self,
        specs: Sequence[RolloutSpec],
        max_concurrency: int,
    ) -> List[RLGroup]:
        results: List[Optional[RLGroup]] = [None] * len(specs)
        next_index = 0
        first_error: Optional[BaseException] = None
        lock = asyncio.Lock()
        stop = asyncio.Event()

        async def worker():
            nonlocal next_index, first_error

            while True:
                async with lock:
                    if stop.is_set() or next_index >= len(specs):
                        return
                    index = next_index
                    next_index += 1

                try:
                    results[index] = await self._rollouts_async(specs[index])
                except Exception as exc:
                    if not stop.is_set():
                        first_error = exc
                        stop.set()
                    return

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(max_concurrency, len(specs)))
        ]
        await asyncio.gather(*tasks)

        if first_error is not None:
            raise first_error

        return [result for result in results if result is not None]

    def train_step(
        self,
        groups: Sequence[object],
        lr: float = 0.002,
    ) -> TrainStepOutput:
        if not groups:
            raise ValueError("train_step requires at least one group")

        payload_groups = []
        for group in groups:
            if isinstance(group, RLGroup):
                if group.rewards is None:
                    raise ValueError("RLGroup rewards must be set before train_step")
                request_payload = group._request_payload
                if request_payload is None:
                    request_payload = self._serialize_request(group.request)
                payload_groups.append(
                    {
                        "mode": "rl",
                        "request": request_payload,
                        "rollouts": group.rollouts,
                        "rewards": group.rewards,
                    }
                )
                continue

            if isinstance(group, SFTGroup):
                request_payload = self._serialize_request(group.request)
                payload_groups.append(
                    {
                        "mode": "sft",
                        "request": request_payload,
                        "targets": self._serialize_targets(
                            request_payload, group.targets
                        ),
                    }
                )
                continue

            raise ValueError("train_step groups must be RLGroup or SFTGroup")

        payload = {
            "finetune_id": self.finetune_id,
            "groups": payload_groups,
            "lr": lr,
        }
        result = self._request_json("POST", "/train_step", payload=payload)
        return {key: result[key] for key in _TRAIN_STEP_OUTPUT_KEYS if key in result}

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

    def download_checkpoint(self, step: int) -> CheckpointDownload:
        return self._request_json(
            "GET", f"/finetunes/{self.finetune_id}/checkpoints/{step}/download"
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


def _validate_name(name: str):
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("name must use only alphanumeric characters, hyphens, or underscores")


def _validate_rank(rank: int):
    if rank not in {8, 16, 24, 32}:
        raise ValueError("rank must be one of 8, 16, 24, or 32")


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

    _validate_name(name)
    _validate_rank(rank)

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
