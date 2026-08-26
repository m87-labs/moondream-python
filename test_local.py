import sys
import os
import asyncio
import threading
from types import SimpleNamespace

import moondream as md
from PIL import Image
from moondream.photon_vl import PhotonStream, PhotonVL, _parse_model


def test_model_parser_preserves_registered_repository_ids():
    assert _parse_model("Qwen/Qwen3.5-4B") == ("Qwen/Qwen3.5-4B", None)
    assert _parse_model("google/gemma-4-E2B-it") == (
        "google/gemma-4-E2B-it",
        None,
    )


def test_model_parser_extracts_explicit_finetune_suffix():
    assert _parse_model("moondream3-preview/01HXYZ@1000") == (
        "moondream3-preview",
        "01HXYZ@1000",
    )
    assert _parse_model("moondream3-preview/ft_abc@1000") == (
        "moondream3-preview",
        "ft_abc@1000",
    )


def test_model_parser_rejects_malformed_finetune_suffixes():
    assert _parse_model("moondream3-preview/01HXYZ") == (
        "moondream3-preview/01HXYZ",
        None,
    )
    assert _parse_model("moondream3-preview/01HXYZ@latest") == (
        "moondream3-preview/01HXYZ@latest",
        None,
    )


def test_vl_local_delegates_to_photon(monkeypatch):
    captured = {}

    def fake_photon(model, *, api_key=None, **runtime_config):
        captured.update(
            model=model,
            api_key=api_key,
            runtime_config=runtime_config,
        )
        return "photon"

    monkeypatch.setattr(md, "photon", fake_photon)

    result = md.vl(
        api_key="key",
        local=True,
        model="Qwen/Qwen3.5-4B",
        max_batch_size=8,
    )

    assert result == "photon"
    assert captured == {
        "model": "Qwen/Qwen3.5-4B",
        "api_key": "key",
        "runtime_config": {"max_batch_size": 8},
    }


def test_vl_local_requires_model():
    try:
        md.vl(local=True)
    except TypeError as exc:
        assert str(exc) == "vl(local=True) requires model=<model identifier>"
    else:
        raise AssertionError("local Photon inference accepted no model")


def test_photon_requires_model():
    try:
        md.photon()
    except TypeError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("Photon accepted no model")


def test_photon_chat_preserves_model_reasoning_default():
    calls = []

    class FakeModel:
        async def chat(self, **prompt):
            calls.append(prompt)
            return SimpleNamespace(
                output={
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": None,
                },
                finish_reason=None,
            )

    client = object.__new__(PhotonVL)
    client._adapter = None
    client._model = FakeModel()
    client._run = asyncio.run

    result = client.chat([{"role": "user", "content": "question"}])
    client.chat(
        [{"role": "user", "content": "question"}],
        reasoning=False,
    )

    assert "reasoning" not in calls[0]
    assert calls[1]["reasoning"] is False
    assert result["finish_reason"] == "stop"


def test_photon_transcribe_passes_options_and_returns_public_output():
    calls = []

    class FakeModel:
        model_id = "openai/whisper-large-v3-turbo"
        tasks = ("transcribe",)

        def supports(self, task):
            return task in self.tasks

        async def transcribe(self, **prompt):
            calls.append(prompt)
            return SimpleNamespace(
                output={
                    "text": "hello world",
                    "language": "en",
                    "segments": [],
                }
            )

    client = object.__new__(PhotonVL)
    client._adapter = None
    client._model = FakeModel()
    client._run = asyncio.run

    result = client.transcribe(
        audio=b"encoded audio",
        task="translate",
        timestamps="word",
        initial_prompt="Moondream, Photon",
        settings={"temperature": 0.0},
    )

    assert result == {
        "text": "hello world",
        "language": "en",
        "segments": [],
    }
    assert calls == [
        {
            "audio": b"encoded audio",
            "task": "translate",
            "timestamps": "word",
            "initial_prompt": "Moondream, Photon",
            "settings": {"temperature": 0.0},
        }
    ]


def test_photon_invoke_rejects_unadvertised_capability():
    class FakeModel:
        model_id = "vision-model"
        tasks = ("query",)

        def supports(self, task):
            return task in self.tasks

    client = object.__new__(PhotonVL)
    client._model = FakeModel()

    try:
        client.invoke("transcribe", audio=b"audio")
    except ValueError as exc:
        assert "does not support 'transcribe'" in str(exc)
    else:
        raise AssertionError("Photon invoked an unadvertised capability")


def test_photon_transcription_stream_preserves_snapshots_and_result():
    class FakeCapabilityStream:
        def __init__(self):
            self._updates = iter(
                [
                    {"text": "hello", "segments": []},
                    {"text": "hello world", "segments": []},
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                output = next(self._updates)
            except StopIteration:
                raise StopAsyncIteration from None
            return SimpleNamespace(output=output)

        async def result(self):
            return SimpleNamespace(
                output={"text": "hello world", "segments": []}
            )

        async def aclose(self):
            return None

    class FakeModel:
        model_id = "openai/whisper-large-v3-turbo"
        tasks = ("transcribe",)

        def supports(self, task):
            return task in self.tasks

        async def transcribe(self, **prompt):
            assert prompt == {"audio": b"audio", "stream": True}
            return FakeCapabilityStream()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    client = object.__new__(PhotonVL)
    client._adapter = None
    client._loop = loop
    client._model = FakeModel()
    try:
        stream = client.transcribe(audio=b"audio", stream=True)
        assert isinstance(stream, PhotonStream)
        assert list(stream) == [
            {"text": "hello", "segments": []},
            {"text": "hello world", "segments": []},
        ]
        assert stream.result() == {"text": "hello world", "segments": []}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_photon_run_returns_single_pass_public_output():
    class FakeModel:
        async def run(self, task, inputs):
            assert task == "embed"
            assert inputs == {"text": "hello"}
            return SimpleNamespace(output={"embedding": [1.0, 2.0]})

    client = object.__new__(PhotonVL)
    client._model = FakeModel()
    client._run = asyncio.run

    assert client.run("embed", {"text": "hello"}) == {
        "embedding": [1.0, 2.0]
    }


def test_photon_stateful_stream_forwards_chunks_updates_and_close():
    sent = []

    class FakeModelStream:
        def __init__(self):
            self._emitted = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._emitted:
                raise StopAsyncIteration
            self._emitted = True
            return SimpleNamespace(output={"text": "partial"})

        async def send(self, **chunk):
            sent.append(chunk)

        async def close(self):
            return SimpleNamespace(output={"text": "final"})

        async def result(self):
            return SimpleNamespace(output={"text": "final"})

    class FakeModel:
        async def stream(self, task, **initial_prompt):
            assert task == "listen"
            assert initial_prompt == {"sample_rate": 16_000}
            return FakeModelStream()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    client = object.__new__(PhotonVL)
    client._loop = loop
    client._model = FakeModel()
    try:
        stream = client.stream("listen", sample_rate=16_000)
        stream.send(audio=b"pcm")
        assert next(stream) == {"text": "partial"}
        assert stream.close() == {"text": "final"}
        assert sent == [{"audio": b"pcm"}]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_photon_live_audio_bridges_application_and_worker_loops():
    source_loops = []

    class FakeLiveStream:
        def __init__(self, audio):
            self._audio = audio.__aiter__()
            self._chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                chunk = await self._audio.__anext__()
            except StopAsyncIteration:
                raise StopAsyncIteration from None
            self._chunks.append(chunk)
            return SimpleNamespace(output={"chunk": chunk})

        async def result(self):
            return SimpleNamespace(output={"chunks": list(self._chunks)})

        async def aclose(self):
            closer = getattr(self._audio, "aclose", None)
            if closer is not None:
                await closer()

    class FakeModel:
        model_id = "openai/whisper-large-v3-turbo"
        tasks = ("transcribe",)

        def supports(self, task):
            return task in self.tasks

        async def transcribe(self, **prompt):
            assert prompt["sample_rate"] == 16_000
            assert prompt["stream"] is True
            return FakeLiveStream(prompt["audio"])

    worker_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=worker_loop.run_forever)
    thread.start()
    client = object.__new__(PhotonVL)
    client._adapter = None
    client._loop = worker_loop
    client._model = FakeModel()

    async def run():
        application_loop = asyncio.get_running_loop()

        async def audio_chunks():
            source_loops.append(asyncio.get_running_loop())
            yield b"first"
            await asyncio.sleep(0)
            source_loops.append(asyncio.get_running_loop())
            yield b"second"

        stream = client.transcribe(
            audio=audio_chunks(),
            sample_rate=16_000,
            stream=True,
        )
        updates = [update async for update in stream]
        assert updates == [{"chunk": b"first"}, {"chunk": b"second"}]
        assert await stream.aresult() == {"chunks": [b"first", b"second"]}
        assert source_loops == [application_loop, application_loop]

    try:
        asyncio.run(run())
    finally:
        worker_loop.call_soon_threadsafe(worker_loop.stop)
        thread.join()
        worker_loop.close()


def test_photon_nonstream_live_audio_uses_async_entrypoint():
    source_loops = []

    class FakeModel:
        model_id = "openai/whisper-large-v3-turbo"
        tasks = ("transcribe",)

        def supports(self, task):
            return task in self.tasks

        async def transcribe(self, **prompt):
            chunks = [chunk async for chunk in prompt["audio"]]
            return SimpleNamespace(output={"chunks": chunks})

    worker_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=worker_loop.run_forever)
    thread.start()
    client = object.__new__(PhotonVL)
    client._adapter = None
    client._loop = worker_loop
    client._model = FakeModel()

    async def run():
        application_loop = asyncio.get_running_loop()

        async def audio_chunks():
            source_loops.append(asyncio.get_running_loop())
            yield b"first"
            await asyncio.sleep(0)
            source_loops.append(asyncio.get_running_loop())
            yield b"second"

        try:
            client.transcribe(audio=audio_chunks(), stream=False)
        except RuntimeError as exc:
            assert "await speech.atranscribe" in str(exc)
        else:
            raise AssertionError("sync live transcription should reject this loop")

        result = await client.atranscribe(
            audio=audio_chunks(),
            stream=False,
        )
        assert result == {"chunks": [b"first", b"second"]}
        assert source_loops == [application_loop, application_loop]

    try:
        asyncio.run(run())
    finally:
        worker_loop.call_soon_threadsafe(worker_loop.stop)
        thread.join()
        worker_loop.close()


def test_photon_stream_fails_after_its_worker_loop_stops():
    class FakeStream:
        async def __anext__(self):
            return SimpleNamespace(output={"text": "unreachable"})

        async def result(self):
            return SimpleNamespace(output={"text": "unreachable"})

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    stream = PhotonStream(FakeStream(), loop)
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    try:
        next(stream)
    except RuntimeError as exc:
        assert "engine is closed" in str(exc)
    else:
        raise AssertionError("stream operation waited on a stopped worker")
    finally:
        loop.close()


def test_photon_stream_detects_worker_stop_after_submission():
    class FakeStream:
        async def __anext__(self):
            return SimpleNamespace(output={"text": "unreachable"})

    loop = asyncio.new_event_loop()
    worker_entered = threading.Event()
    release_worker = threading.Event()
    submission_queued = threading.Event()
    outcome = []

    def block_then_stop():
        worker_entered.set()
        release_worker.wait()
        loop.stop()

    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    loop.call_soon_threadsafe(block_then_stop)
    assert worker_entered.wait(timeout=1)

    original_call_soon = loop.call_soon_threadsafe

    def observed_call_soon(callback, *args, **kwargs):
        handle = original_call_soon(callback, *args, **kwargs)
        submission_queued.set()
        return handle

    loop.call_soon_threadsafe = observed_call_soon
    stream = PhotonStream(FakeStream(), loop)

    def consume():
        try:
            next(stream)
        except BaseException as exc:
            outcome.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert submission_queued.wait(timeout=1)
    release_worker.set()
    thread.join(timeout=1)
    consumer.join(timeout=1)
    loop.call_soon_threadsafe = original_call_soon
    try:
        assert not consumer.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], RuntimeError)
        assert "engine is closed" in str(outcome[0])
    finally:
        if consumer.is_alive():
            rescue = threading.Thread(target=loop.run_forever)
            rescue.start()
            consumer.join(timeout=1)
            original_call_soon(loop.stop)
            rescue.join(timeout=1)
        loop.close()


def test_photon_stream_propagates_worker_timeout_sync_and_async():
    class TimeoutStream:
        async def __anext__(self):
            raise TimeoutError("operation timeout")

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        sync_stream = PhotonStream(TimeoutStream(), loop)
        try:
            next(sync_stream)
        except TimeoutError as exc:
            assert str(exc) == "operation timeout"
        else:
            raise AssertionError("sync stream swallowed the worker timeout")

        async def consume():
            async_stream = PhotonStream(TimeoutStream(), loop)
            try:
                await async_stream.__anext__()
            except TimeoutError as exc:
                assert str(exc) == "operation timeout"
            else:
                raise AssertionError("async stream swallowed the worker timeout")

        asyncio.run(consume())
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def main(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path '{image_path}'")
        sys.exit(1)

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Instantiate the client in local mode.
    client = md.vl(local=True, model="moondream3.1-9B-A2B")

    # Test the caption method.
    try:
        print("Starting caption")
        caption_output = client.caption(image, settings={"max_tokens": 10})
        print("Caption:", caption_output.get("caption"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption test failed:", e)

    try:
        print("Starting caption stream")
        for chunk in client.caption(image, length="long", stream=True)["caption"]:
            print(chunk, end="", flush=True)
        print("\n------ done ------\n")
    except Exception as e:
        print("Caption stream test failed:", e)

    try:
        print("start Query")
        query_output = client.query(
            image, "What's in the image?", settings={"max_tokens": 10}
        )
        print("Query Answer:", query_output.get("answer"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Query test failed:", e)

    try:
        print("Starting query stream")
        for chunk in client.query(image, "What's in the image?", stream=True)["answer"]:
            print(chunk, end="", flush=True)
        print("\n------ done ------\n")
    except Exception as e:
        print("query stream test failed:", e)

    # Test the detect method.
    try:
        print("Starting output")
        detect_output = client.detect(image, "item")
        print("Detected Objects:", detect_output.get("objects"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Detect test failed:", e)

    # Test the point method.
    try:
        print("Starting Point")
        point_output = client.point(image, "person")
        print("Points:", point_output.get("points"))
        print("\n------ done ------\n")
    except Exception as e:
        print("Point test failed:", e)


if __name__ == "__main__":
    image_path = "moondream/assets/how-to-be-a-people-person-1662995088.jpg"
    main(image_path)
