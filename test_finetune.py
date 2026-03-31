import json
import socket
import threading
import unittest
import urllib.error
from unittest import mock

import moondream as md
from PIL import Image

from moondream.finetune import Finetune, FinetuneAPIError, ft
from moondream.types import EncodedImage, RLGroup, RolloutRequest, SFTGroup


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _http_error(status, body, headers=None):
    if isinstance(body, dict):
        body = json.dumps(body).encode("utf-8")
    error = urllib.error.HTTPError(
        url="https://example.test",
        code=status,
        msg="error",
        hdrs=headers or {},
        fp=mock.Mock(read=mock.Mock(return_value=body)),
    )
    error.close = mock.Mock()
    return error


def _raw_rollout(skill, output):
    return {
        "skill": skill,
        "finish_reason": "stop",
        "output": output,
        "answer_tokens": [1],
        "thinking_tokens": [],
        "has_answer_separator": False,
        "coords": [],
        "sizes": [],
    }


class FinetuneTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4, 4), color="white")
        self.client = Finetune(
            api_key="test-key",
            endpoint="https://api.example.test/v1/tuning",
            finetune_id="ft_123",
            name="demo-ft",
            rank=8,
            max_retries=2,
            retry_base_delay=0.01,
            retry_max_delay=0.01,
            timeout=0.01,
        )

    def test_ft_validates_constructor_inputs(self):
        with self.assertRaises(ValueError):
            ft(api_key="x")

        with self.assertRaises(ValueError):
            ft(api_key="x", name="demo", finetune_id="ft_123")

        with self.assertRaises(ValueError):
            ft(api_key="x", name="demo", rank=8, max_retries=-1)

    def test_package_exposes_helper_types_under_md_types(self):
        self.assertTrue(hasattr(md, "ft"))
        self.assertTrue(hasattr(md, "types"))
        self.assertIs(md.types.RolloutRequest, RolloutRequest)
        self.assertIs(md.types.RLGroup, RLGroup)
        self.assertIs(md.types.SFTGroup, SFTGroup)
        self.assertFalse(hasattr(md, "RolloutRequest"))
        self.assertFalse(hasattr(md, "RLGroup"))
        self.assertFalse(hasattr(md, "SFTGroup"))

    def test_ft_binds_existing_finetune(self):
        response = {
            "finetune": {
                "finetune_id": "ft_456",
                "name": "existing-ft",
                "rank": 16,
            }
        }

        with mock.patch.object(Finetune, "_request_json", return_value=response) as mocked:
            client = ft(
                api_key="x",
                finetune_id="ft_456",
                endpoint="https://api.example.test/v1/tuning",
            )

        self.assertEqual(client.finetune_id, "ft_456")
        self.assertEqual(client.name, "existing-ft")
        self.assertEqual(client.rank, 16)
        mocked.assert_called_once_with("GET", "/finetunes/ft_456")

    def test_ft_creates_new_finetune(self):
        with mock.patch.object(
            Finetune,
            "_request_json",
            return_value={"finetune_id": "ft_789"},
        ) as mocked:
            client = ft(
                api_key="x",
                name="new-ft",
                rank=12,
                endpoint="https://api.example.test/v1/tuning",
            )

        self.assertEqual(client.finetune_id, "ft_789")
        self.assertEqual(client.name, "new-ft")
        self.assertEqual(client.rank, 12)
        mocked.assert_called_once_with(
            "POST",
            "/finetunes",
            payload={"name": "new-ft", "rank": 12},
        )

    def test_rollouts_serializes_request_and_returns_raw_response(self):
        response = {
            "request": {
                "skill": "query",
                "question": "What is happening?",
                "image_url": "data:image/jpeg;base64,abc",
                "reasoning": True,
                "settings": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
            },
            "rollouts": [_raw_rollout("query", {"answer": "People are socializing."})],
        }

        request = RolloutRequest(
            skill="query",
            image=self.image,
            question="What is happening?",
            num_rollouts=2,
            reasoning=True,
            settings={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
        )

        with mock.patch.object(self.client, "_request_json", return_value=response) as mocked:
            result = self.client.rollouts(request)

        self.assertEqual(result, response)
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["finetune_id"], "ft_123")
        self.assertEqual(payload["num_rollouts"], 2)
        self.assertEqual(payload["request"]["skill"], "query")
        self.assertEqual(payload["request"]["question"], "What is happening?")
        self.assertTrue(payload["request"]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(payload["request"]["reasoning"])

    def test_rollouts_pass_settings_through(self):
        request = RolloutRequest(
            skill="query",
            question="What is here?",
            image=self.image,
            settings={"max_objects": 2},
        )

        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"request": {"skill": "query"}, "rollouts": []},
        ) as mocked:
            self.client.rollouts(request)

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["request"]["settings"]["max_objects"], 2)

    def test_rollouts_pass_ground_truth_through(self):
        request = RolloutRequest(
            skill="detect",
            image=self.image,
            object="vehicles",
            ground_truth={"boxes": []},
        )

        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"request": {"skill": "detect"}, "rollouts": []},
        ) as mocked:
            self.client.rollouts(request)

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["ground_truth"], {"boxes": []})

    def test_rollouts_reject_unknown_encoded_image(self):
        class FakeEncodedImage(EncodedImage):
            pass

        with self.assertRaises(ValueError):
            self.client.rollouts(
                RolloutRequest(skill="detect", image=FakeEncodedImage(), object="vehicles")
            )

    def test_batch_rollouts_returns_rl_groups(self):
        async def fake_rollouts_async(request):
            return {
                "request": {
                    "skill": request.skill,
                    "question": request.question,
                },
                "rollouts": [_raw_rollout(request.skill, {"answer": request.question})],
                "rewards": [0.5],
            }

        requests = [
            RolloutRequest(skill="query", question="q0"),
            RolloutRequest(skill="query", question="q1"),
        ]

        with mock.patch.object(self.client, "_rollouts_async", side_effect=fake_rollouts_async):
            groups = self.client.batch_rollouts(requests, max_concurrency=2)

        self.assertEqual(groups[0]["mode"], "rl")
        self.assertEqual(groups[0]["request"]["question"], "q0")
        self.assertEqual(groups[0]["rollouts"][0]["output"]["answer"], "q0")
        self.assertEqual(groups[0]["rewards"], [0.5])

    def test_batch_rollouts_validates_max_concurrency(self):
        with self.assertRaises(ValueError):
            self.client.batch_rollouts([], max_concurrency=0)

    def test_sft_group_builds_http_shaped_group(self):
        group = self.client.sft_group(
            skill="query",
            image=self.image,
            question="What is happening?",
            targets=[{"answer": "People are smiling for a photo."}],
            reasoning=True,
        )

        self.assertEqual(group["mode"], "sft")
        self.assertEqual(group["request"]["skill"], "query")
        self.assertEqual(group["request"]["question"], "What is happening?")
        self.assertTrue(group["request"]["reasoning"])
        self.assertEqual(group["targets"], [{"answer": "People are smiling for a photo."}])
        self.assertTrue(group["request"]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_train_step_builds_mixed_rl_and_sft_payload(self):
        raw_rollout = _raw_rollout("query", {"answer": "A sign"})
        rl_group: RLGroup = {
            "mode": "rl",
            "request": {
                "skill": "query",
                "question": "What is this?",
                "image_url": "data:image/jpeg;base64,abc",
            },
            "rollouts": [raw_rollout],
            "rewards": [1.0],
        }
        sft_group = self.client.sft_group(
            skill="query",
            image=self.image,
            question="What country is this?",
            targets=[{"answer": "United States"}],
        )

        with mock.patch.object(
            self.client, "_request_json", return_value={"step": 1, "applied": True}
        ) as mocked:
            result = self.client.train_step([rl_group, sft_group], lr=0.003)

        self.assertEqual(result["step"], 1)
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["finetune_id"], "ft_123")
        self.assertEqual(payload["lr"], 0.003)
        self.assertEqual(payload["groups"][0]["mode"], "rl")
        self.assertEqual(payload["groups"][0]["rollouts"], [raw_rollout])
        self.assertEqual(payload["groups"][0]["rewards"], [1.0])
        self.assertEqual(payload["groups"][1]["mode"], "sft")
        self.assertEqual(payload["groups"][1]["targets"], [{"answer": "United States"}])

    def test_train_step_hides_internal_metrics(self):
        rl_group: RLGroup = {
            "mode": "rl",
            "request": {"skill": "query", "question": "What is this?"},
            "rollouts": [_raw_rollout("query", {"answer": "A photo"})],
            "rewards": [1.0],
        }

        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"step": 1, "applied": True, "internal_metric": 0.5},
        ):
            result = self.client.train_step([rl_group])

        self.assertEqual(result, {"step": 1, "applied": True})

    def test_log_metrics_builds_payload_and_returns_response(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"ok": True, "step": 100, "logged_count": 2},
        ) as mocked:
            result = self.client.log_metrics(
                100,
                {
                    "eval/country_match": 0.63,
                    "eval/token_f1": 0.64,
                },
            )

        self.assertEqual(result, {"ok": True, "step": 100, "logged_count": 2})
        mocked.assert_called_once_with(
            "POST",
            "/finetunes/ft_123/metrics",
            payload={
                "step": 100,
                "metrics": {
                    "eval/country_match": 0.63,
                    "eval/token_f1": 0.64,
                },
            },
        )

    def test_log_metrics_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.client.log_metrics(-1, {"eval/score": 1.0})

        with self.assertRaises(ValueError):
            self.client.log_metrics(1, {})

        with self.assertRaises(ValueError):
            self.client.log_metrics(1, {"bad name": 1.0})

        with self.assertRaises(ValueError):
            self.client.log_metrics(1, {"sys/loss": 1.0})

        with self.assertRaises(ValueError):
            self.client.log_metrics(1, {"eval/score": float("nan")})

        too_many = {f"eval/m{i}": float(i) for i in range(101)}
        with self.assertRaises(ValueError):
            self.client.log_metrics(1, too_many)

    def test_public_rollout_to_train_step_handoff(self):
        rollout_response = {
            "request": {
                "skill": "query",
                "question": "What is happening?",
                "image_url": "data:image/jpeg;base64,abc",
            },
            "rollouts": [_raw_rollout("query", {"answer": "People are socializing."})],
        }

        with mock.patch.object(
            self.client,
            "_request_json",
            side_effect=[rollout_response, {"step": 4, "applied": True}],
        ) as mocked:
            response = self.client.rollouts(
                RolloutRequest(
                    skill="query",
                    image=self.image,
                    question="What is happening?",
                )
            )
            group: RLGroup = {
                "mode": "rl",
                "request": response["request"],
                "rollouts": response["rollouts"],
                "rewards": [1.0],
            }
            result = self.client.train_step([group])

        self.assertEqual(result["step"], 4)
        train_payload = mocked.call_args_list[1].kwargs["payload"]
        self.assertEqual(train_payload["groups"][0]["request"], rollout_response["request"])
        self.assertEqual(train_payload["groups"][0]["rollouts"], rollout_response["rollouts"])
        self.assertEqual(train_payload["groups"][0]["rewards"], [1.0])

    def test_train_step_rejects_rl_group_without_rewards(self):
        group: RLGroup = {
            "mode": "rl",
            "request": {"skill": "query", "question": "What is this?"},
            "rollouts": [_raw_rollout("query", {"answer": "A photo"})],
        }

        with self.assertRaises(ValueError):
            self.client.train_step([group])

    def test_train_step_rejects_mutated_rewards_length(self):
        group: RLGroup = {
            "mode": "rl",
            "request": {"skill": "query", "question": "What is this?"},
            "rollouts": [
                _raw_rollout("query", {"answer": "A photo"}),
                _raw_rollout("query", {"answer": "A drawing"}),
            ],
            "rewards": [1.0],
        }

        with self.assertRaises(ValueError):
            self.client.train_step([group])

    def test_request_json_retries_timeout_then_succeeds(self):
        attempts = {"count": 0}

        def urlopen(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise urllib.error.URLError(socket.timeout("timed out"))
            return _FakeResponse({"ok": True})

        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            with mock.patch("time.sleep"):
                result = self.client._request_json("GET", "/health")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts["count"], 3)

    def test_request_json_does_not_retry_bad_api_key(self):
        error = _http_error(401, {"error": "invalid api key"})
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=error,
        ) as mocked:
            with self.assertRaises(FinetuneAPIError) as ctx:
                self.client._request_json("GET", "/finetunes/ft_123")

        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.body, "invalid api key")
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(mocked.call_count, 1)
        error.close.assert_called_once()

    def test_request_json_retries_524_then_succeeds(self):
        errors = [_http_error(524, "error code: 524"), _http_error(524, "error code: 524")]

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[errors[0], errors[1], _FakeResponse({"ok": True})],
        ):
            with mock.patch("time.sleep"):
                result = self.client._request_json("POST", "/rollouts", payload={"x": 1})

        self.assertEqual(result, {"ok": True})
        for error in errors:
            error.close.assert_called_once()

    def test_train_step_does_not_retry_timeout(self):
        group: RLGroup = {
            "mode": "rl",
            "request": {"skill": "query", "question": "What is this?"},
            "rollouts": [_raw_rollout("query", {"answer": "A photo"})],
            "rewards": [1.0],
        }

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(socket.timeout("timed out")),
        ) as mocked:
            with mock.patch("time.sleep") as mocked_sleep:
                with self.assertRaises(FinetuneAPIError):
                    self.client.train_step([group])

        self.assertEqual(mocked.call_count, 1)
        mocked_sleep.assert_not_called()

    def test_batch_rollouts_preserves_order_and_concurrency_cap(self):
        active = {"count": 0, "max": 0}

        async def fake_rollouts_async(request):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await __import__("asyncio").sleep(0.01)
            active["count"] -= 1
            return {
                "request": {"skill": request.skill, "question": request.question},
                "rollouts": [],
            }

        requests = [RolloutRequest(skill="query", question=f"q{i}") for i in range(5)]

        with mock.patch.object(self.client, "_rollouts_async", side_effect=fake_rollouts_async):
            groups = self.client.batch_rollouts(requests, max_concurrency=2)

        self.assertEqual(
            [group["request"]["question"] for group in groups],
            [f"q{i}" for i in range(5)],
        )
        self.assertLessEqual(active["max"], 2)

    def test_batch_rollouts_drains_in_flight_requests_after_failure(self):
        client = Finetune(
            api_key="test-key",
            endpoint="https://api.example.test/v1/tuning",
            finetune_id="ft_123",
            name="demo-ft",
            rank=8,
            max_retries=0,
            retry_base_delay=0.01,
            retry_max_delay=0.01,
            timeout=0.01,
        )
        q0_started = threading.Event()
        q1_started = threading.Event()
        q2_started = threading.Event()
        release_q0 = threading.Event()
        started = []
        result = {}

        def request_json_once(method, path, payload=None, query=None):
            question = payload["request"]["question"]
            started.append(question)

            if question == "q0":
                q0_started.set()
                self.assertTrue(release_q0.wait(timeout=1))
                return {"request": payload["request"], "rollouts": []}

            if question == "q1":
                q1_started.set()
                self.assertTrue(q0_started.wait(timeout=1))
                raise urllib.error.URLError("boom")

            q2_started.set()
            return {"request": payload["request"], "rollouts": []}

        def run_batch_rollouts():
            try:
                client.batch_rollouts(
                    [
                        RolloutRequest(skill="query", question="q0"),
                        RolloutRequest(skill="query", question="q1"),
                        RolloutRequest(skill="query", question="q2"),
                    ],
                    max_concurrency=2,
                )
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=run_batch_rollouts)

        with mock.patch.object(client, "_request_json_once", side_effect=request_json_once):
            worker.start()
            self.assertTrue(q0_started.wait(timeout=1))
            self.assertTrue(q1_started.wait(timeout=1))
            worker.join(timeout=0.05)
            self.assertTrue(worker.is_alive())
            self.assertFalse(q2_started.is_set())

            release_q0.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result.get("error"), FinetuneAPIError)
        self.assertEqual(len(started), 2)
        self.assertCountEqual(started, ["q0", "q1"])

    def test_list_checkpoints_validates_limit(self):
        with self.assertRaises(ValueError):
            self.client.list_checkpoints(limit=0)


if __name__ == "__main__":
    unittest.main()
