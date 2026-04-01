import json
import socket
import threading
import time
import unittest
import urllib.error
from unittest import mock

import moondream as md
from PIL import Image

from moondream.finetune import Finetune, ft
from moondream.types import EncodedImage, RLGroup, SFTGroup


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
        )

    def test_ft_validates_constructor_inputs(self):
        with self.assertRaises(ValueError):
            ft(api_key="x")

        with self.assertRaises(ValueError):
            ft(api_key="x", name="demo", finetune_id="ft_123")


    def test_package_exposes_helper_types_under_md_types(self):
        self.assertTrue(hasattr(md, "ft"))
        self.assertTrue(hasattr(md, "types"))
        self.assertIs(md.types.RLGroup, RLGroup)
        self.assertIs(md.types.SFTGroup, SFTGroup)
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

        with mock.patch.object(self.client, "_request_json", return_value=response) as mocked:
            result = self.client.rollouts(
                "query",
                image=self.image,
                question="What is happening?",
                num_rollouts=2,
                reasoning=True,
                settings={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
            )

        self.assertEqual(result, response)
        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["finetune_id"], "ft_123")
        self.assertEqual(payload["num_rollouts"], 2)
        self.assertEqual(payload["request"]["skill"], "query")
        self.assertEqual(payload["request"]["question"], "What is happening?")
        self.assertTrue(payload["request"]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(payload["request"]["reasoning"])

    def test_rollouts_pass_settings_through(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"request": {"skill": "query"}, "rollouts": []},
        ) as mocked:
            self.client.rollouts(
                "query",
                question="What is here?",
                image=self.image,
                settings={"max_objects": 2},
            )

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["request"]["settings"]["max_objects"], 2)

    def test_rollouts_pass_ground_truth_through(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"request": {"skill": "detect"}, "rollouts": []},
        ) as mocked:
            self.client.rollouts(
                "detect",
                image=self.image,
                object="vehicles",
                ground_truth={"boxes": []},
            )

        payload = mocked.call_args.kwargs["payload"]
        self.assertEqual(payload["ground_truth"], {"boxes": []})

    def test_rollouts_reject_unknown_encoded_image(self):
        class FakeEncodedImage(EncodedImage):
            pass

        with self.assertRaises(ValueError):
            self.client.rollouts("detect", image=FakeEncodedImage(), object="vehicles")

    def test_rollout_stream_yields_context_response_pairs(self):
        def fake_rollouts(skill, **kwargs):
            question = kwargs.get("question", "")
            return {
                "request": {
                    "skill": skill,
                    "question": question,
                },
                "rollouts": [_raw_rollout(skill, {"answer": question})],
                "rewards": [0.5],
            }

        items = [
            ({"label": "rock"}, {"skill": "query", "question": "q0"}),
            ({"label": "paper"}, {"skill": "query", "question": "q1"}),
        ]

        with mock.patch.object(self.client, "rollouts", side_effect=fake_rollouts):
            results = list(self.client.rollout_stream(items, max_concurrency=2))

        self.assertEqual(len(results), 2)
        contexts = {r[0]["label"] for r in results}
        self.assertEqual(contexts, {"rock", "paper"})
        for context, response in results:
            self.assertIn("request", response)
            self.assertIn("rollouts", response)

    def test_rollout_stream_validates_params(self):
        with self.assertRaises(ValueError):
            list(self.client.rollout_stream([], max_concurrency=0))
        with self.assertRaises(ValueError):
            list(self.client.rollout_stream([], buffer_size=0))

    def test_train_step_encodes_images_in_groups(self):
        sft_group: SFTGroup = {
            "mode": "sft",
            "request": {
                "skill": "query",
                "question": "What is happening?",
                "image": self.image,
            },
            "target": {"answer": "People are smiling."},
        }

        with mock.patch.object(
            self.client, "_request_json", return_value={"step": 1, "applied": True}
        ) as mocked:
            self.client.train_step([sft_group])

        payload = mocked.call_args.kwargs["payload"]
        request = payload["groups"][0]["request"]
        self.assertNotIn("image", request)
        self.assertTrue(request["image_url"].startswith("data:image/jpeg;base64,"))

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
        sft_group: SFTGroup = {
            "mode": "sft",
            "request": {
                "skill": "query",
                "question": "What country is this?",
                "image": self.image,
            },
            "target": {"answer": "United States"},
        }

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
        self.assertEqual(payload["groups"][1]["target"], {"answer": "United States"})
        self.assertTrue(payload["groups"][1]["request"]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_train_step_returns_raw_response(self):
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

        self.assertEqual(result, {"step": 1, "applied": True, "internal_metric": 0.5})

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

    def test_log_metrics_passes_inputs_through_without_local_validation(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"ok": True},
        ) as mocked:
            self.client.log_metrics(
                -1,
                {
                    "bad name": 1.0,
                    "sys/loss": 2.0,
                },
            )

        mocked.assert_called_once_with(
            "POST",
            "/finetunes/ft_123/metrics",
            payload={
                "step": -1,
                "metrics": {
                    "bad name": 1.0,
                    "sys/loss": 2.0,
                },
            },
        )

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
                "query",
                image=self.image,
                question="What is happening?",
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

    def test_train_step_passes_groups_through_without_local_validation(self):
        group: RLGroup = {
            "mode": "rl",
            "request": {"skill": "query", "question": "What is this?"},
            "rollouts": [_raw_rollout("query", {"answer": "A photo"})],
        }

        with mock.patch.object(
            self.client, "_request_json", return_value={"step": 7, "applied": True}
        ) as mocked:
            result = self.client.train_step([group])

        self.assertEqual(result["step"], 7)
        mocked.assert_called_once_with(
            "POST",
            "/train_step",
            payload={
                "finetune_id": "ft_123",
                "groups": [group],
                "lr": 0.002,
            },
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
        error = _http_error(401, {"error": "invalid api key"})
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=error,
        ) as mocked:
            with self.assertRaises(urllib.error.HTTPError):
                self.client._request_json("GET", "/finetunes/ft_123")

        self.assertEqual(mocked.call_count, 1)

    def test_request_json_retries_524_then_succeeds(self):
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=[
                _http_error(524, "error code: 524"),
                _http_error(524, "error code: 524"),
                _FakeResponse({"ok": True}),
            ],
        ):
            with mock.patch("time.sleep"):
                result = self.client._request_json("POST", "/rollouts", payload={"x": 1})

        self.assertEqual(result, {"ok": True})

    def test_rollout_stream_respects_concurrency_cap(self):
        active = {"count": 0, "max": 0}
        lock = threading.Lock()

        def fake_rollouts(skill, **kwargs):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.02)
            with lock:
                active["count"] -= 1
            return {
                "request": {"skill": skill, "question": kwargs.get("question", "")},
                "rollouts": [],
            }

        items = [(i, {"skill": "query", "question": f"q{i}"}) for i in range(5)]

        with mock.patch.object(self.client, "rollouts", side_effect=fake_rollouts):
            results = list(self.client.rollout_stream(items, max_concurrency=2))

        self.assertEqual(len(results), 5)
        self.assertLessEqual(active["max"], 2)

    def test_rollout_stream_stops_on_error(self):
        call_count = [0]

        def fake_rollouts(skill, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            time.sleep(0.02)
            return {
                "request": {"skill": skill, "question": kwargs.get("question", "")},
                "rollouts": [],
            }

        items = [(i, {"skill": "query", "question": f"q{i}"}) for i in range(10)]

        with mock.patch.object(self.client, "rollouts", side_effect=fake_rollouts):
            with self.assertRaises(RuntimeError):
                list(self.client.rollout_stream(items, max_concurrency=2))

    def test_rollout_stream_stops_before_extra_requests_on_error(self):
        """stop.set() must happen before blocking on the queue so other
        workers don't pull additional requests after the first failure."""
        started = []
        lock = threading.Lock()

        def fake_rollouts(skill, **kwargs):
            question = kwargs.get("question", "")
            with lock:
                started.append(question)
            if question == "q1":
                raise RuntimeError("boom")
            time.sleep(0.1)
            return {
                "request": {"skill": skill, "question": question},
                "rollouts": [],
            }

        items = [(i, {"skill": "query", "question": f"q{i}"}) for i in range(20)]

        with mock.patch.object(self.client, "rollouts", side_effect=fake_rollouts):
            with self.assertRaises(RuntimeError):
                list(self.client.rollout_stream(
                    items, max_concurrency=2, buffer_size=1,
                ))

        # Workers should not have started many requests beyond the failure
        self.assertLess(len(started), 6)

    def test_rollout_stream_surfaces_iterator_error(self):
        def bad_iterator():
            yield (None, {"skill": "query", "question": "q0"})
            raise RuntimeError("dataset failed")

        def fake_rollouts(skill, **kwargs):
            time.sleep(0.02)
            return {
                "request": {"skill": skill, "question": kwargs.get("question", "")},
                "rollouts": [],
            }

        with mock.patch.object(self.client, "rollouts", side_effect=fake_rollouts):
            with self.assertRaises(RuntimeError) as ctx:
                list(self.client.rollout_stream(bad_iterator(), max_concurrency=2))
            self.assertIn("dataset failed", str(ctx.exception))

    def test_list_checkpoints_pass_limit_through_without_local_validation(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"checkpoints": [], "has_more": False},
        ) as mocked:
            self.client.list_checkpoints(limit=0)

        mocked.assert_called_once_with(
            "GET",
            "/finetunes/ft_123/checkpoints",
            query={"limit": 0, "cursor": None},
        )

    def test_save_checkpoint_returns_raw_response(self):
        response = {
            "ok": True,
            "checkpoint": {
                "checkpoint_id": "ckpt_123",
                "finetune_id": "ft_123",
                "step": 7,
            },
        }

        with mock.patch.object(
            self.client,
            "_request_json",
            return_value=response,
        ) as mocked:
            result = self.client.save_checkpoint()

        self.assertEqual(result, response)
        mocked.assert_called_once_with(
            "POST",
            "/finetunes/ft_123/checkpoints/save",
        )

    def test_delete_returns_none(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"ok": True},
        ) as mocked:
            result = self.client.delete()

        self.assertIsNone(result)
        mocked.assert_called_once_with("DELETE", "/finetunes/ft_123")

    def test_delete_checkpoint_returns_none(self):
        with mock.patch.object(
            self.client,
            "_request_json",
            return_value={"ok": True},
        ) as mocked:
            result = self.client.delete_checkpoint(7)

        self.assertIsNone(result)
        mocked.assert_called_once_with(
            "DELETE",
            "/finetunes/ft_123/checkpoints/7",
        )


if __name__ == "__main__":
    unittest.main()
