from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class InsightToolInput(BaseModel):
    """Input schema for InsightTool."""

    transcript: str = Field(..., description="Whole call transcript")
    summary: str = Field(..., description="Executive summary text")
    sentiments: str = Field(..., description="Sentiment JSON list")


class InsightTool(BaseTool):
    """Extract pain points, compliance issues, escalation triggers as JSON."""

    name: str = "InsightTool"
    description: str = (
        "Given transcript, summary, and sentiment analysis, return JSON with "
        "keys pain_points, compliance_issues, escalation_triggers (each list of strings)."
    )
    args_schema: Type[BaseModel] = InsightToolInput

    def _run(self, transcript: str, summary: str, sentiments: str) -> str:  # type: ignore[override]
        logger.debug("Running InsightTool with len transcript %s", len(transcript))
        try:
            import openai  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError("openai package is required for InsightTool") from exc

        system_prompt = (
            "You analyse customer-service call data to find actionable insights. "
            "Output strict JSON: {pain_points:[], compliance_issues:[], escalation_triggers:[]}."
        )
        user_msg = (
            "TRANSCRIPT:\n" + transcript + "\n\nSUMMARY:\n" + summary + "\n\nSENTIMENTS:\n" + sentiments
        )
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()
