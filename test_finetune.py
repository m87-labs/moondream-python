import json
import socket
import threading
import unittest
import urllib.error
from unittest import mock

from PIL import Image

import moondream as md
from moondream.finetune import Finetune, FinetuneAPIError, ft
from moondream.types import EncodedImage, RLGroup, RolloutSpec, SFTGroup


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
    return urllib.error.HTTPError(
        url="https://example.test",
        code=status,
        msg="error",
        hdrs=headers or {},
        fp=mock.Mock(read=mock.Mock(return_value=body)),
    )


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
            ft(api_key="x", name="bad name", rank=8)

        with self.assertRaises(ValueError):
            ft(api_key="x", name="demo", rank=12)

        with self.assertRaises(ValueError):
            ft(api_key="x", name="demo", rank=8, max_retries=-1)

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
                rank=8,
                endpoint="https://api.example.test/v1/tuning",
            )

        self.assertEqual(client.finetune_id, "ft_789")
        self.assertEqual(client.name, "new-ft")
        self.assertEqual(client.rank, 8)
        mocked.assert_called_once_with(
            "POST",
            "/finetunes",
            payload={"name": "new-ft", "rank": 8},
        )

    def test_rollouts_serializes_request_and_returns_rl_group(self):
        request = {
            "skill": "detect",
            "object": "vehicles",
            "image": self.image,
            "settings": {"temperature": 1.0, "top_p": 1.0, "max_objects": 5},
        }
        response = {
            "request": {
                "skill": "detect",
                "object": "vehicles",
                "image_url": "data:image/jpeg;base64,abc",
            },
            "rollouts": [
                {
                    "skill": "detect",
                    "finish_reason": "stop",
                    "output": {"objects": []},
                    "answer_tokens": [],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
            "rewards": None,
        }

        with mock.patch.object(self.client, "_request_json", return_value=response) as mocked:
            rl_group = self.client.rollouts(request, num_rollouts=2)

        self.assertIsInstance(rl_group, RLGroup)
        self.assertEqual(rl_group.request, request)
        self.assertEqual(len(rl_group.rollouts), 1)

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["finetune_id"], "ft_123")
        self.assertEqual(payload["num_rollouts"], 2)
        self.assertEqual(payload["request"]["skill"], "detect")
        self.assertTrue(payload["request"]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_rollouts_rejects_query_ground_truth(self):
        with self.assertRaises(ValueError):
            self.client.rollouts(
                {"skill": "query", "question": "What is here?", "image": self.image},
                ground_truth={"boxes": []},
            )

    def test_rollouts_reject_unknown_encoded_image(self):
        class FakeEncodedImage(EncodedImage):
            pass

        with self.assertRaises(ValueError):
            self.client.rollouts(
                {
                    "skill": "detect",
                    "object": "vehicles",
                    "image": FakeEncodedImage(),
                }
            )

    def test_rlgroup_with_rewards_validates_length(self):
        rl_group = RLGroup(
            request={"skill": "query", "question": "hi"},
            rollouts=[
                {
                    "skill": "query",
                    "finish_reason": "stop",
                    "output": {"answer": "hello"},
                    "answer_tokens": [],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
        )

        with self.assertRaises(ValueError):
            rl_group.with_rewards([1.0, 0.0])

    def test_train_step_builds_mixed_rl_and_sft_payload(self):
        rl_group = RLGroup(
            request={"skill": "query", "question": "What is this?", "image": self.image},
            rollouts=[
                {
                    "skill": "query",
                    "finish_reason": "stop",
                    "output": {"answer": "A sign"},
                    "answer_tokens": [1],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
            rewards=[1.0],
            _request_payload={
                "skill": "query",
                "question": "What is this?",
                "image_url": "data:image/jpeg;base64,abc",
            },
        )
        sft_group = SFTGroup(
            request={
                "skill": "query",
                "question": "What country is this?",
                "image": self.image,
                "reasoning": False,
                "settings": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16},
            },
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
        self.assertEqual(payload["groups"][0]["rewards"], [1.0])
        self.assertEqual(payload["groups"][1]["mode"], "sft")
        self.assertEqual(payload["groups"][1]["targets"], [{"answer": "United States"}])
        self.assertIn("image_url", payload["groups"][1]["request"])

    def test_train_step_hides_internal_metrics(self):
        rl_group = RLGroup(
            request={"skill": "query", "question": "What is this?"},
            rollouts=[
                {
                    "skill": "query",
                    "finish_reason": "stop",
                    "output": {"answer": "A photo"},
                    "answer_tokens": [1],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
            rewards=[1.0],
        )

        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"step": 1, "applied": True, "internal_metric": 0.5},
        ):
            result = self.client.train_step([rl_group])

        self.assertEqual(result, {"step": 1, "applied": True})

    def test_public_rollout_to_train_step_handoff(self):
        rollout_response = {
            "request": {
                "skill": "query",
                "question": "What is happening?",
                "image_url": "data:image/jpeg;base64,abc",
            },
            "rollouts": [
                {
                    "skill": "query",
                    "finish_reason": "stop",
                    "output": {"answer": "People are socializing."},
                    "answer_tokens": [1],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
        }

        with mock.patch.object(
            self.client,
            "_request_json",
            side_effect=[rollout_response, {"step": 4, "applied": True}],
        ) as mocked:
            rl_group = self.client.rollouts(
                {
                    "skill": "query",
                    "question": "What is happening?",
                    "image": self.image,
                }
            )
            result = self.client.train_step([rl_group.with_rewards([1.0])])

        self.assertEqual(result["step"], 4)
        train_payload = mocked.call_args_list[1].kwargs["payload"]
        self.assertEqual(
            train_payload["groups"][0]["request"],
            rollout_response["request"],
        )
        self.assertEqual(train_payload["groups"][0]["rewards"], [1.0])

    def test_train_step_serializes_manual_rl_group_request(self):
        rl_group = RLGroup(
            request={
                "skill": "detect",
                "object": "vehicles",
                "image": self.image,
                "settings": {"temperature": 0.7, "top_p": 0.9, "max_objects": 4},
            },
            rollouts=[
                {
                    "skill": "detect",
                    "finish_reason": "stop",
                    "output": {"objects": []},
                    "answer_tokens": [],
                    "thinking_tokens": [],
                    "has_answer_separator": False,
                    "coords": [],
                    "sizes": [],
                }
            ],
            rewards=[0.5],
        )

        with mock.patch.object(
            self.client, "_request_json", return_value={"step": 2, "applied": True}
        ) as mocked:
            result = self.client.train_step([rl_group])

        self.assertEqual(result["step"], 2)
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["groups"][0]["mode"], "rl")
        self.assertEqual(payload["groups"][0]["rewards"], [0.5])
        self.assertEqual(payload["groups"][0]["request"]["skill"], "detect")
        self.assertEqual(payload["groups"][0]["request"]["object"], "vehicles")
        self.assertEqual(payload["groups"][0]["request"]["settings"]["max_objects"], 4)
        self.assertIn("image_url", payload["groups"][0]["request"])

    def test_detect_rollouts_allow_empty_boxes_ground_truth(self):
        response = {
            "request": {
                "skill": "detect",
                "object": "vehicles",
                "image_url": "data:image/jpeg;base64,abc",
            },
            "rollouts": [],
        }

        with mock.patch.object(self.client, "_request_json", return_value=response) as mocked:
            self.client.rollouts(
                {
                    "skill": "detect",
                    "object": "vehicles",
                    "image": self.image,
                },
                ground_truth={"boxes": []},
            )

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["ground_truth"], {"boxes": []})

    def test_detect_sft_targets_allow_empty_boxes(self):
        group = SFTGroup(
            request={
                "skill": "detect",
                "object": "vehicles",
                "image": self.image,
            },
            targets=[{"boxes": []}],
        )

        with mock.patch.object(
            self.client, "_request_json", return_value={"step": 5, "applied": True}
        ) as mocked:
            result = self.client.train_step([group])

        self.assertEqual(result["step"], 5)
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["groups"][0]["targets"], [{"boxes": []}])

    def test_query_sft_target_validation(self):
        with self.assertRaises(ValueError):
            self.client.train_step(
                [
                    SFTGroup(
                        request={
                            "skill": "query",
                            "question": "Why?",
                            "image": self.image,
                            "reasoning": True,
                        },
                        targets=[{"answer": "Because"}],
                    )
                ]
            )

        with self.assertRaises(ValueError):
            self.client.train_step(
                [
                    SFTGroup(
                        request={
                            "skill": "query",
                            "question": "Why?",
                            "image": self.image,
                            "reasoning": False,
                        },
                        targets=[{"answer": "Because", "reasoning": {"text": "x", "grounding": []}}],
                    )
                ]
            )

        with self.assertRaises(ValueError):
            self.client.train_step(
                [
                    SFTGroup(
                        request={"skill": "query", "question": "Why?"},
                        targets=[],
                    )
                ]
            )

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
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=_http_error(401, {"error": "invalid api key"}),
        ) as mocked:
            with self.assertRaises(FinetuneAPIError) as ctx:
                self.client._request_json("GET", "/finetunes/ft_123")

        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.body, "invalid api key")
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(mocked.call_count, 1)

    def test_request_json_retries_524_then_succeeds(self):
        attempts = {"count": 0}

        def urlopen(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise _http_error(524, "error code: 524")
            return _FakeResponse({"ok": True})

        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            with mock.patch("time.sleep"):
                result = self.client._request_json("POST", "/rollouts", payload={"x": 1})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts["count"], 3)

    def test_rollout_groups_preserves_order_and_concurrency_cap(self):
        active = {"count": 0, "max": 0}

        async def fake_rollouts_async(spec):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await __import__("asyncio").sleep(0.01)
            active["count"] -= 1
            return RLGroup(
                request=spec.request,
                rollouts=[],
                rewards=[1.0],
                _request_payload={"skill": spec.request["skill"]},
            )

        specs = [
            RolloutSpec(request={"skill": "query", "question": f"q{i}"}, num_rollouts=1)
            for i in range(5)
        ]

        with mock.patch.object(self.client, "_rollouts_async", side_effect=fake_rollouts_async):
            rl_groups = self.client.rollout_groups(specs, max_concurrency=2)

        self.assertEqual([group.request["question"] for group in rl_groups], [f"q{i}" for i in range(5)])
        self.assertLessEqual(active["max"], 2)

    def test_rollout_groups_drains_in_flight_requests_after_failure(self):
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

        def run_rollout_groups():
            try:
                client.rollout_groups(
                    [
                        RolloutSpec(request={"skill": "query", "question": "q0"}),
                        RolloutSpec(request={"skill": "query", "question": "q1"}),
                        RolloutSpec(request={"skill": "query", "question": "q2"}),
                    ],
                    max_concurrency=2,
                )
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=run_rollout_groups)

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
