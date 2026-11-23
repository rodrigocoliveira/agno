import base64
import mimetypes
from dataclasses import dataclass, field
from os import getenv
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from agno.exceptions import ModelAuthenticationError, ModelProviderError
from agno.media import File
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.utils.log import log_error, log_warning
from agno.utils.openai import audio_to_message, images_to_message

# DeepSeek file ids look like "file-api-...". agno File ids are auto-generated UUIDs, so this
# prefix is what marks a File as already uploaded to DeepSeek.
DEEPSEEK_FILE_ID_PREFIX = "file-api-"

# Documents DeepSeek only reads after parsing them on upload (purpose "text_recognition" or
# "rich_layout_recognition"). Other non-image files (txt, md, csv, json, xlsx...) use "user_data".
# See https://api-docs.deepseek.com/guides/files_api
DEEPSEEK_PARSED_DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@dataclass
class DeepSeek(OpenAILike):
    """
    A class for interacting with DeepSeek models.

    DeepSeek V4 models (deepseek-v4-flash, deepseek-v4-pro) run with thinking mode
    enabled by default. While thinking mode is active, the following parameters are
    accepted but have no effect (they are silently ignored by the API):
    - temperature
    - top_p
    - presence_penalty
    - frequency_penalty

    Thinking mode is controlled with the ``use_thinking`` flag:
    - ``use_thinking=None`` (default): thinking is on for thinking-capable models and
      off for the legacy non-thinking ids (deepseek-chat).
    - ``use_thinking=True``: force thinking on.
    - ``use_thinking=False``: turn thinking off (faster, cheaper responses).

    For agent scenarios, DeepSeek recommends ``reasoning_effort="max"``. It is left
    unset (None) by default so the API uses its own default ("high"); set it
    explicitly to opt in. Valid values are "high" and "max".

    For more information, see: https://api-docs.deepseek.com/guides/thinking_mode

    Attributes:
        id (str): The model id. Defaults to "deepseek-v4-flash".
        name (str): The model name. Defaults to "DeepSeek".
        provider (str): The provider name. Defaults to "DeepSeek".
        api_key (Optional[str]): The API key.
        base_url (str): The base URL. Defaults to "https://api.deepseek.com".
        use_thinking (Optional[bool]): Toggle thinking mode. None uses the model default.
        document_upload_purpose (str): Files API purpose used for PDF and Office documents.
            "text_recognition" (default) extracts text only; "rich_layout_recognition" also keeps figures.

    File input follows DeepSeek's format, not OpenAI's: images are sent inline as a flat
    ``{"type": "file", "file_data": ...}`` part, and every other file is uploaded through the
    Files API and referenced as ``{"type": "file", "file_id": ...}``. The DeepSeek file id is
    written back to ``File.id`` so a file replayed from history is not uploaded again; a File
    created with an existing ``file-api-...`` id is sent as is.
    See: https://api-docs.deepseek.com/guides/document_parsing/
    """

    id: str = "deepseek-v4-flash"
    name: str = "DeepSeek"
    provider: str = "DeepSeek"

    api_key: Optional[str] = field(default_factory=lambda: getenv("DEEPSEEK_API_KEY"))
    base_url: str = "https://api.deepseek.com"

    # Toggle thinking mode. None = use the model default (on for thinking-capable models,
    # off for the legacy non-thinking ids). True = force on, False = force off.
    use_thinking: Optional[bool] = None

    # Files API purpose for PDF and Office documents: "text_recognition" or "rich_layout_recognition".
    document_upload_purpose: str = "text_recognition"

    # DeepSeek supports JSON mode (response_format={"type": "json_object"}) but not
    # native/json_schema structured outputs, so output_schema needs use_json_mode=True.
    supports_native_structured_outputs: bool = False

    # Model ids that should NOT have thinking enabled by default. The legacy ids still
    # work and route server-side: deepseek-chat -> non-thinking mode of deepseek-v4-flash,
    # deepseek-reasoner -> thinking mode of deepseek-v4-flash. Only deepseek-chat needs to
    # opt out of thinking here (deepseek-reasoner is already a thinking model).
    _non_thinking_model_ids: ClassVar[Set[str]] = {
        "deepseek-chat",
    }

    def _thinking_enabled(self) -> bool:
        """Resolve whether thinking mode should be on for this request.

        An explicit ``use_thinking`` flag always wins; otherwise thinking-capable models
        default to on and the legacy non-thinking ids default to off.
        """
        if self.use_thinking is not None:
            return self.use_thinking
        return self.id not in self._non_thinking_model_ids

    def get_request_params(
        self,
        response_format=None,
        tools=None,
        tool_choice=None,
        run_response=None,
    ) -> Dict[str, Any]:
        request_params = super().get_request_params(
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
        )

        if self._thinking_enabled():
            # Merge with any user-supplied extra_body and never overwrite an explicit
            # thinking setting (so a raw extra_body override still takes precedence).
            extra_body = request_params.get("extra_body") or {}
            extra_body.setdefault("thinking", {"type": "enabled"})
            request_params["extra_body"] = extra_body
        else:
            # No thinking: reasoning_effort has no effect, so strip it. Only send an
            # explicit disabled flag when the user turned a thinking-capable model off;
            # the legacy non-thinking ids are already non-thinking server-side.
            request_params.pop("reasoning_effort", None)
            if self.use_thinking is False:
                extra_body = request_params.get("extra_body") or {}
                extra_body.setdefault("thinking", {"type": "disabled"})
                request_params["extra_body"] = extra_body

        return request_params

    def _get_client_params(self) -> Dict[str, Any]:
        # Fetch API key from env if not already set
        if not self.api_key:
            self.api_key = getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                # Raise error immediately if key is missing
                raise ModelAuthenticationError(
                    message="DEEPSEEK_API_KEY not set. Please set the DEEPSEEK_API_KEY environment variable.",
                    model_name=self.name,
                )

        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add additional client params if provided
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    @staticmethod
    def _file_name_and_mime(file: File) -> Tuple[str, Optional[str]]:
        """Derive the upload filename and mime type of a File."""
        filename = file.filename
        if not filename and file.filepath is not None:
            filename = Path(str(file.filepath)).name
        if not filename and file.url is not None:
            filename = Path(urlparse(file.url).path).name
        filename = filename or "file"
        mime_type = file.mime_type or mimetypes.guess_type(filename)[0]
        # DeepSeek detects the document type from the filename, so give extension-less names one.
        if mime_type and not Path(filename).suffix:
            filename += mimetypes.guess_extension(mime_type) or ""
        return filename, mime_type

    def _upload_kwargs(self, filename: str, data: bytes, mime_type: Optional[str]) -> Dict[str, Any]:
        """Build the Files API arguments, picking the purpose DeepSeek expects for the file type."""
        if mime_type is None or mime_type in DEEPSEEK_PARSED_DOCUMENT_MIME_TYPES:
            purpose = self.document_upload_purpose
        else:
            purpose = "user_data"
        file_tuple = (filename, data, mime_type) if mime_type else (filename, data)
        return {"file": file_tuple, "purpose": purpose}

    def _upload_error(self, filename: str, error: Exception) -> ModelProviderError:
        return ModelProviderError(
            message=f"Failed to upload file '{filename}' to DeepSeek: {error}",
            model_name=self.name,
            model_id=self.id,
        )

    def _format_file(self, file: File) -> Optional[Dict[str, Any]]:
        """Build a DeepSeek file part, uploading the file first when it is not an image.

        DeepSeek's file part is flat ({"type": "file", "file_id": ...}), not OpenAI's nested
        {"type": "file", "file": {...}}. Inline file_data only accepts images; documents must
        be uploaded through the Files API and referenced by file_id.
        """
        if file.id and file.id.startswith(DEEPSEEK_FILE_ID_PREFIX):
            return {"type": "file", "file_id": file.id}

        filename, mime_type = self._file_name_and_mime(file)
        try:
            data = file.get_content_bytes()
        except Exception as e:
            log_error(f"Failed to read file '{filename}': {e}")
            return None
        if not data:
            log_error(f"No content to send for file '{filename}'.")
            return None

        if mime_type and mime_type.startswith("image/"):
            encoded = base64.b64encode(data).decode("utf-8")
            return {"type": "file", "file_data": f"data:{mime_type};base64,{encoded}", "filename": filename}

        try:
            result = self.get_client().files.create(**self._upload_kwargs(filename, data, mime_type))
        except Exception as e:
            raise self._upload_error(filename, e) from e
        file.id = result.id
        return {"type": "file", "file_id": result.id}

    async def _aupload_files(self, messages: List[Message]) -> None:
        """Upload pending non-image files with the async client before the request is built.

        Message formatting is synchronous, so without this the async path would block the event
        loop on each upload. Uploaded files get their DeepSeek id, which _format_file then sends as is.
        """
        for message in messages:
            for file in message.files or []:
                if file.id and file.id.startswith(DEEPSEEK_FILE_ID_PREFIX):
                    continue
                filename, mime_type = self._file_name_and_mime(file)
                if mime_type and mime_type.startswith("image/"):
                    continue
                try:
                    data = await file.aget_content_bytes()
                except Exception:
                    data = None
                if not data:
                    # Left to _format_file, which logs and skips unreadable files.
                    continue
                try:
                    result = await self.get_async_client().files.create(
                        **self._upload_kwargs(filename, data, mime_type)
                    )
                except Exception as e:
                    raise self._upload_error(filename, e) from e
                file.id = result.id

    async def ainvoke(self, messages: List[Message], *args: Any, **kwargs: Any) -> ModelResponse:
        await self._aupload_files(messages)
        return await super().ainvoke(messages, *args, **kwargs)

    async def ainvoke_stream(  # type: ignore[override]
        self, messages: List[Message], *args: Any, **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        await self._aupload_files(messages)
        async for response in super().ainvoke_stream(messages, *args, **kwargs):
            yield response

    def _format_message(self, message: Message, compress_tool_results: bool = False) -> Dict[str, Any]:
        """
        Format a message into the format expected by OpenAI.

        Args:
            message (Message): The message to format.
            compress_tool_results: Whether to compress tool results.

        Returns:
            Dict[str, Any]: The formatted message.
        """
        tool_result = message.get_content(use_compressed_content=compress_tool_results)

        message_dict: Dict[str, Any] = {
            "role": self.role_map[message.role] if self.role_map else self.default_role_map[message.role],
            "content": tool_result,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": message.tool_calls,
            "reasoning_content": message.reasoning_content,
        }
        message_dict = {k: v for k, v in message_dict.items() if v is not None}

        # Ignore non-string message content
        # because we assume that the images/audio are already added to the message
        if (message.images is not None and len(message.images) > 0) or (
            message.audio is not None and len(message.audio) > 0
        ):
            # Ignore non-string message content
            # because we assume that the images/audio are already added to the message
            if isinstance(message.content, str):
                message_dict["content"] = [{"type": "text", "text": message.content}]
                if message.images is not None:
                    message_dict["content"].extend(images_to_message(images=message.images))

                if message.audio is not None:
                    message_dict["content"].extend(audio_to_message(audio=message.audio))

        if message.audio_output is not None:
            message_dict["content"] = ""
            message_dict["audio"] = {"id": message.audio_output.id}

        if message.videos is not None and len(message.videos) > 0:
            log_warning("Video input is currently unsupported.")

        if message.files is not None:
            # Ensure content is a list of parts
            content = message_dict.get("content")
            if isinstance(content, str):  # wrap existing text
                text = content
                message_dict["content"] = [{"type": "text", "text": text}]
            elif content is None:
                message_dict["content"] = []
            # Insert each file part before text parts
            for file in message.files:
                file_part = self._format_file(file)
                if file_part:
                    message_dict["content"].insert(0, file_part)

        # Manually add the content field even if it is None
        if message.content is None:
            message_dict["content"] = ""
        return message_dict
