"""FastAPI wrapper exposing a single /analyze endpoint for call analytics."""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
import base64
import logging
from call_analytics.crew import CallAnalytics

logger = logging.getLogger(__name__)
app = FastAPI(title="Call Analytics API")

# simple static HTML UI
@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:  # noqa: D401
    return FileResponse(__file__.replace("api.py", "static/index.html"))


@app.post("/analyze")
async def analyze_call(file: UploadFile = File(...)):
    """Analyze an audio file and return structured insights."""
    try:
        contents = await file.read()
        audio_b64 = base64.b64encode(contents).decode()
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed reading file")
        raise HTTPException(status_code=400, detail="Invalid file upload") from exc

    # Kick off the crew with input variables expected by transcribe tool
    result = CallAnalytics().crew().kickoff(
        inputs={
            "audio_base64": audio_b64,
            "file_name": file.filename or "audio.wav",
        }
    )
    return {"result": result}
