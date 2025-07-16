from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class SentimentToolInput(BaseModel):
    """Input schema for SentimentTool."""

    text: str = Field(..., description="Speaker turn text to analyse")


class SentimentTool(BaseTool):
    """LLM-based sentiment analysis returning JSON."""

    name: str = "sentiment_analysis"
    description: str = "Analyse text sentiment and return JSON {sentiment, score}."
    args_schema: Type[BaseModel] = SentimentToolInput

    def _run(self, text: str) -> str:  # type: ignore[override]
        logger.debug("Running SentimentTool for text length %s", len(text))

        try:
            import openai  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError("openai package is required for SentimentTool") from exc

        client = openai.OpenAI()
        system_prompt = (
            "You are a sentiment classifier. "
            "Output JSON exactly with keys 'sentiment' (positive|neutral|negative) and 'score' (float -1..1)."
        )
        user_msg = f"Text: {text}"
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
