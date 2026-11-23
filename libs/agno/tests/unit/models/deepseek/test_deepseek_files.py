"""File input for DeepSeek: flat file parts, Files API uploads for documents, inline images."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.exceptions import ModelProviderError
from agno.media import File
from agno.models.deepseek import DeepSeek
from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat

PDF = b"%PDF-1.3 fake pdf"
PNG = b"\x89PNG\r\n\x1a\n fake png"


def _model_with_client(file_id: str = "file-api-abc"):
    model = DeepSeek(api_key="test")
    client = MagicMock()
    client.files.create.return_value = SimpleNamespace(id=file_id)
    return model, client


def _file_parts(message_dict):
    return [p for p in message_dict["content"] if p.get("type") == "file"]


def test_pdf_is_uploaded_and_sent_as_flat_file_id():
    model, client = _model_with_client()
    file = File(content=PDF, mime_type="application/pdf")
    message = Message(role="user", content="What is the price?", files=[file])

    with patch.object(model, "get_client", return_value=client):
        message_dict = model._format_message(message)

    assert _file_parts(message_dict) == [{"type": "file", "file_id": "file-api-abc"}]
    kwargs = client.files.create.call_args.kwargs
    assert kwargs["purpose"] == "text_recognition"
    # Extension-less names get one from the mime type, since DeepSeek reads the type from the filename.
    assert kwargs["file"] == ("file.pdf", PDF, "application/pdf")
    assert file.id == "file-api-abc"


def test_file_with_deepseek_id_is_not_uploaded_again():
    model, client = _model_with_client()
    file = File(id="file-api-existing", content=PDF, mime_type="application/pdf")

    with patch.object(model, "get_client", return_value=client):
        message_dict = model._format_message(Message(role="user", content="q", files=[file]))

    assert _file_parts(message_dict) == [{"type": "file", "file_id": "file-api-existing"}]
    client.files.create.assert_not_called()


def test_auto_generated_file_id_is_not_sent_as_deepseek_id():
    model, client = _model_with_client()
    file = File(content=PDF, mime_type="application/pdf")
    assert file.id and not file.id.startswith("file-api-")

    with patch.object(model, "get_client", return_value=client):
        model._format_message(Message(role="user", content="q", files=[file]))

    client.files.create.assert_called_once()


def test_image_file_is_sent_inline_as_flat_file_data():
    model, client = _model_with_client()
    file = File(content=PNG, filename="red.png")

    with patch.object(model, "get_client", return_value=client):
        message_dict = model._format_message(Message(role="user", content="q", files=[file]))

    [part] = _file_parts(message_dict)
    assert part["file_data"].startswith("data:image/png;base64,")
    assert part["filename"] == "red.png"
    assert "file" not in part
    client.files.create.assert_not_called()


def test_plain_text_file_uses_user_data_purpose():
    model, client = _model_with_client()
    file = File(content=b"hello", mime_type="text/plain", filename="notes.txt")

    with patch.object(model, "get_client", return_value=client):
        model._format_message(Message(role="user", content="q", files=[file]))

    assert client.files.create.call_args.kwargs["purpose"] == "user_data"


def test_document_upload_purpose_is_configurable():
    model, client = _model_with_client()
    model.document_upload_purpose = "rich_layout_recognition"
    file = File(content=PDF, mime_type="application/pdf")

    with patch.object(model, "get_client", return_value=client):
        model._format_message(Message(role="user", content="q", files=[file]))

    assert client.files.create.call_args.kwargs["purpose"] == "rich_layout_recognition"


def test_upload_failure_raises_model_provider_error():
    model, client = _model_with_client()
    client.files.create.side_effect = RuntimeError("boom")
    file = File(content=PDF, mime_type="application/pdf")

    with patch.object(model, "get_client", return_value=client):
        with pytest.raises(ModelProviderError, match="Failed to upload file 'file.pdf' to DeepSeek: boom"):
            model._format_message(Message(role="user", content="q", files=[file]))


async def test_async_path_uploads_with_async_client():
    model = DeepSeek(api_key="test")
    async_client = MagicMock()
    async_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-api-async"))
    sync_client = MagicMock()
    file = File(content=PDF, mime_type="application/pdf")
    messages = [Message(role="user", content="q", files=[file])]

    with patch.object(model, "get_async_client", return_value=async_client):
        with patch.object(model, "get_client", return_value=sync_client):
            with patch.object(OpenAIChat, "ainvoke", new_callable=AsyncMock) as parent_ainvoke:
                await model.ainvoke(messages=messages, assistant_message=Message(role="assistant"))
                message_dict = model._format_message(messages[0])

    async_client.files.create.assert_awaited_once()
    sync_client.files.create.assert_not_called()
    parent_ainvoke.assert_awaited_once()
    assert _file_parts(message_dict) == [{"type": "file", "file_id": "file-api-async"}]


async def test_async_stream_path_uploads_before_streaming():
    model = DeepSeek(api_key="test")
    async_client = MagicMock()
    async_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-api-stream"))
    file = File(content=PDF, mime_type="application/pdf")

    async def fake_stream(self, *args, **kwargs):
        assert file.id == "file-api-stream"
        yield "chunk"

    with patch.object(model, "get_async_client", return_value=async_client):
        with patch.object(OpenAIChat, "ainvoke_stream", fake_stream):
            chunks = [
                c
                async for c in model.ainvoke_stream(
                    messages=[Message(role="user", content="q", files=[file])],
                    assistant_message=Message(role="assistant"),
                )
            ]

    assert chunks == ["chunk"]
    async_client.files.create.assert_awaited_once()
