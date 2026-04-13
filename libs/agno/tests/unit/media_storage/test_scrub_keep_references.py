"""Tests for scrub_media_from_run_output with keep_references=True.

When store_media=False + media_storage is configured, the scrub should preserve
media that was successfully offloaded (has media_reference) and only remove
raw-content media. This is the core behavior that enables multi-turn
conversations to reconstruct images from external storage.
"""

from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock, Mock

from agno.agent.agent import Agent
from agno.media import Audio, File, Image, Video
from agno.media_storage.reference import MediaReference
from agno.models.base import Model
from agno.models.message import Message, MessageMetrics
from agno.models.response import ModelResponse
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunOutput
from agno.utils.agent import scrub_media_from_message, scrub_media_from_run_output

# -- Helpers --


def _make_ref(media_id: str = "ref-1", backend: str = "local") -> MediaReference:
    return MediaReference(
        media_id=media_id,
        storage_key=f"agno/media/{media_id}.png",
        storage_backend=backend,
        url=f"https://storage.example.com/{media_id}",
    )


def _offloaded_image(media_id: str = "img-offloaded") -> Image:
    """Image that was offloaded to external storage (has media_reference, no content)."""
    ref = _make_ref(media_id)
    return Image(url=ref.url, media_reference=ref, id=media_id)


def _raw_image(media_id: str = "img-raw") -> Image:
    """Image with raw content bytes (NOT offloaded)."""
    return Image(content=b"fake-png-bytes", id=media_id, mime_type="image/png")


def _offloaded_audio(media_id: str = "aud-offloaded") -> Audio:
    ref = _make_ref(media_id)
    return Audio(url=ref.url, media_reference=ref, id=media_id)


def _raw_audio(media_id: str = "aud-raw") -> Audio:
    return Audio(content=b"fake-audio", id=media_id)


def _offloaded_video(media_id: str = "vid-offloaded") -> Video:
    ref = _make_ref(media_id)
    return Video(url=ref.url, media_reference=ref, id=media_id)


def _raw_video(media_id: str = "vid-raw") -> Video:
    return Video(content=b"fake-video", id=media_id)


def _offloaded_file(media_id: str = "file-offloaded") -> File:
    ref = _make_ref(media_id)
    return File(url=ref.url, media_reference=ref, id=media_id)


def _raw_file(media_id: str = "file-raw") -> File:
    return File(content=b"fake-file", id=media_id)


def _mock_storage():
    """Mock MediaStorage backend."""
    storage = MagicMock()
    storage.backend_name = "mock"
    storage.bucket = "test-bucket"
    storage.region = "us-east-1"
    storage.persist_remote_urls = False
    storage.upload.return_value = "agno/media/test-id.ext"
    storage.get_url.return_value = "https://example.com/presigned-url"
    return storage


class MockModel(Model):
    """Minimal mock model for Agent instantiation."""

    def __init__(self):
        super().__init__(id="test-model", name="test-model", provider="test")
        self.instructions = None
        self._mock_response = ModelResponse(
            content="response text",
            role="assistant",
            response_usage=MessageMetrics(),
        )
        self.response = Mock(return_value=self._mock_response)

    def get_instructions_for_model(self, *args, **kwargs):
        return None

    def get_system_message_for_model(self, *args, **kwargs):
        return None

    async def aget_instructions_for_model(self, *args, **kwargs):
        return None

    async def aget_system_message_for_model(self, *args, **kwargs):
        return None

    def parse_args(self, *args, **kwargs):
        return {}

    def invoke(self, *args, **kwargs) -> ModelResponse:
        return self._mock_response

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        return self._mock_response

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        yield self._mock_response

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        yield self._mock_response
        return

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return self._mock_response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._mock_response


# ---------------------------------------------------------------------------
# scrub_media_from_run_output with keep_references=True
# ---------------------------------------------------------------------------


class TestScrubKeepReferencesRunOutput:
    def test_preserves_offloaded_images_removes_raw(self):
        """Offloaded images (with media_reference) are preserved; raw images are removed."""
        run_output = RunOutput(
            images=[_offloaded_image("img-1"), _raw_image("img-2")],
        )
        scrub_media_from_run_output(run_output, keep_references=True)

        assert run_output.images is not None
        assert len(run_output.images) == 1
        assert run_output.images[0].id == "img-1"
        assert run_output.images[0].media_reference is not None

    def test_preserves_offloaded_videos_removes_raw(self):
        run_output = RunOutput(
            videos=[_offloaded_video("vid-1"), _raw_video("vid-2")],
        )
        scrub_media_from_run_output(run_output, keep_references=True)

        assert run_output.videos is not None
        assert len(run_output.videos) == 1
        assert run_output.videos[0].id == "vid-1"

    def test_preserves_offloaded_audio_removes_raw(self):
        run_output = RunOutput(
            audio=[_offloaded_audio("aud-1"), _raw_audio("aud-2")],
        )
        scrub_media_from_run_output(run_output, keep_references=True)

        assert run_output.audio is not None
        assert len(run_output.audio) == 1
        assert run_output.audio[0].id == "aud-1"

    def test_preserves_offloaded_files_removes_raw(self):
        run_output = RunOutput(
            files=[_offloaded_file("file-1"), _raw_file("file-2")],
        )
        scrub_media_from_run_output(run_output, keep_references=True)

        assert run_output.files is not None
        assert len(run_output.files) == 1
        assert run_output.files[0].id == "file-1"

    def test_nulls_when_no_offloaded_media_remains(self):
        """When all media is raw (no references), result should be None."""
        run_output = RunOutput(
            images=[_raw_image()],
            videos=[_raw_video()],
            audio=[_raw_audio()],
            files=[_raw_file()],
        )
        scrub_media_from_run_output(run_output, keep_references=True)

        assert run_output.images is None
        assert run_output.videos is None
        assert run_output.audio is None
        assert run_output.files is None

    def test_preserves_offloaded_input_media(self):
        """RunInput media with references is preserved."""
        run_input = RunInput(
            input_content="test input",
            images=[_offloaded_image("in-img-1"), _raw_image("in-img-2")],
            videos=[_offloaded_video("in-vid-1")],
            audios=[_raw_audio("in-aud-1")],
            files=[_offloaded_file("in-file-1")],
        )
        run_output = RunOutput(input=run_input)
        scrub_media_from_run_output(run_output, keep_references=True)

        assert len(run_output.input.images) == 1
        assert run_output.input.images[0].id == "in-img-1"
        assert len(run_output.input.videos) == 1
        assert len(run_output.input.audios) == 0
        assert len(run_output.input.files) == 1

    def test_works_on_team_run_output(self):
        team_output = TeamRunOutput(
            images=[_offloaded_image("team-img"), _raw_image("team-raw")],
        )
        scrub_media_from_run_output(team_output, keep_references=True)

        assert team_output.images is not None
        assert len(team_output.images) == 1
        assert team_output.images[0].id == "team-img"


# ---------------------------------------------------------------------------
# scrub_media_from_message with keep_references=True
# ---------------------------------------------------------------------------


class TestScrubKeepReferencesMessage:
    def test_preserves_offloaded_message_images(self):
        msg = Message(role="user", content="test")
        msg.images = [_offloaded_image("msg-img-1"), _raw_image("msg-img-2")]

        scrub_media_from_message(msg, keep_references=True)

        assert msg.images is not None
        assert len(msg.images) == 1
        assert msg.images[0].id == "msg-img-1"

    def test_preserves_offloaded_message_audio(self):
        msg = Message(role="user", content="test")
        msg.audio = [_offloaded_audio("msg-aud-1"), _raw_audio("msg-aud-2")]

        scrub_media_from_message(msg, keep_references=True)

        assert msg.audio is not None
        assert len(msg.audio) == 1
        assert msg.audio[0].id == "msg-aud-1"

    def test_preserves_offloaded_message_videos(self):
        msg = Message(role="user", content="test")
        msg.videos = [_offloaded_video("msg-vid-1"), _raw_video("msg-vid-2")]

        scrub_media_from_message(msg, keep_references=True)

        assert msg.videos is not None
        assert len(msg.videos) == 1
        assert msg.videos[0].id == "msg-vid-1"

    def test_preserves_offloaded_message_files(self):
        msg = Message(role="user", content="test")
        msg.files = [_offloaded_file("msg-file-1"), _raw_file("msg-file-2")]

        scrub_media_from_message(msg, keep_references=True)

        assert msg.files is not None
        assert len(msg.files) == 1
        assert msg.files[0].id == "msg-file-1"

    def test_preserves_offloaded_audio_output(self):
        msg = Message(role="assistant", content="test")
        msg.audio_output = _offloaded_audio("aud-out")

        scrub_media_from_message(msg, keep_references=True)

        assert msg.audio_output is not None
        assert msg.audio_output.id == "aud-out"

    def test_removes_raw_audio_output(self):
        msg = Message(role="assistant", content="test")
        msg.audio_output = _raw_audio("aud-out-raw")

        scrub_media_from_message(msg, keep_references=True)

        assert msg.audio_output is None

    def test_preserves_offloaded_image_output(self):
        msg = Message(role="assistant", content="test")
        msg.image_output = _offloaded_image("img-out")

        scrub_media_from_message(msg, keep_references=True)

        assert msg.image_output is not None

    def test_removes_raw_image_output(self):
        msg = Message(role="assistant", content="test")
        msg.image_output = _raw_image("img-out-raw")

        scrub_media_from_message(msg, keep_references=True)

        assert msg.image_output is None

    def test_nulls_message_media_when_no_references_remain(self):
        msg = Message(role="user", content="test")
        msg.images = [_raw_image()]
        msg.audio = [_raw_audio()]
        msg.videos = [_raw_video()]
        msg.files = [_raw_file()]

        scrub_media_from_message(msg, keep_references=True)

        assert msg.images is None
        assert msg.audio is None
        assert msg.videos is None
        assert msg.files is None


# ---------------------------------------------------------------------------
# Backward compatibility: keep_references=False (default)
# ---------------------------------------------------------------------------


class TestScrubDefaultRemovesEverything:
    def test_default_removes_all_media_including_offloaded(self):
        """Default keep_references=False removes ALL media, even offloaded."""
        run_output = RunOutput(
            images=[_offloaded_image(), _raw_image()],
            videos=[_offloaded_video()],
            audio=[_offloaded_audio()],
            files=[_offloaded_file()],
        )
        scrub_media_from_run_output(run_output)  # keep_references defaults to False

        assert run_output.images is None
        assert run_output.videos is None
        assert run_output.audio is None
        assert run_output.files is None

    def test_default_removes_all_message_media(self):
        msg = Message(role="user", content="test")
        msg.images = [_offloaded_image()]
        msg.audio_output = _offloaded_audio()

        scrub_media_from_message(msg)  # keep_references defaults to False

        assert msg.images is None
        assert msg.audio_output is None


# ---------------------------------------------------------------------------
# scrub_run_output_for_storage integration with Agent flags
# ---------------------------------------------------------------------------


class TestScrubRunOutputForStorage:
    def test_store_media_false_with_media_storage_keeps_references(self):
        """Agent(store_media=False, media_storage=configured) preserves offloaded media."""
        from agno.agent._run import scrub_run_output_for_storage

        agent = Agent(model=MockModel(), store_media=False, media_storage=_mock_storage())

        run_output = RunOutput(
            images=[_offloaded_image("kept"), _raw_image("removed")],
        )
        scrub_run_output_for_storage(agent, run_output)

        assert run_output.images is not None
        assert len(run_output.images) == 1
        assert run_output.images[0].id == "kept"
        assert run_output.images[0].media_reference is not None

    def test_store_media_false_without_media_storage_removes_everything(self):
        """Agent(store_media=False, media_storage=None) removes all media."""
        from agno.agent._run import scrub_run_output_for_storage

        agent = Agent(model=MockModel(), store_media=False)

        run_output = RunOutput(
            images=[_offloaded_image(), _raw_image()],
        )
        scrub_run_output_for_storage(agent, run_output)

        assert run_output.images is None

    def test_store_media_true_preserves_all_media(self):
        """Agent(store_media=True) does not scrub media at all."""
        from agno.agent._run import scrub_run_output_for_storage

        agent = Agent(model=MockModel(), store_media=True)

        offloaded = _offloaded_image("off")
        raw = _raw_image("raw")
        run_output = RunOutput(images=[offloaded, raw])
        scrub_run_output_for_storage(agent, run_output)

        assert run_output.images is not None
        assert len(run_output.images) == 2

    def test_store_media_false_with_media_storage_scrubs_messages(self):
        """Messages are also scrubbed with keep_references when media_storage is set."""
        from agno.agent._run import scrub_run_output_for_storage

        agent = Agent(model=MockModel(), store_media=False, media_storage=_mock_storage())

        msg = Message(role="user", content="test")
        msg.images = [_offloaded_image("msg-kept"), _raw_image("msg-removed")]

        run_output = RunOutput(messages=[msg])
        scrub_run_output_for_storage(agent, run_output)

        assert msg.images is not None
        assert len(msg.images) == 1
        assert msg.images[0].id == "msg-kept"
