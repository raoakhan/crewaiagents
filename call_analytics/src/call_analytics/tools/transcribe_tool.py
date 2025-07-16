from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import base64
import pathlib

class TranscribeToolInput(BaseModel):
    """Input for transcribing audio."""

    audio_base64: str = Field(..., description="Audio file content, base64-encoded WAV/MP3")
    file_name: str = Field(..., description="Original filename for logging")


class TranscribeTool(BaseTool):
    """Tool that sends audio to the OpenAI Whisper API and returns a transcript."""

    name: str = "transcribe_tool"
    description: str = (
        "Convert base64-encoded audio bytes into a text transcript using OpenAI's Whisper endpoint."
    )
    args_schema: Type[BaseModel] = TranscribeToolInput

    def _run(self, audio_base64: str, file_name: str) -> str:  # type: ignore[override]
        try:
            import openai  # noqa: WPS433 external call
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for TranscribeTool") from exc

        audio_bytes = base64.b64decode(audio_base64)
        tmp_path = pathlib.Path("/tmp") / file_name
        tmp_path.write_bytes(audio_bytes)

        client = openai.OpenAI()
        with tmp_path.open("rb") as f:
            transcript_resp = client.audio.transcriptions.create(  # type: ignore[attr-defined]
                model="whisper-1",
                file=f,
                response_format="text",
                language="en"
            )
        tmp_path.unlink(missing_ok=True)
        return transcript_resp
