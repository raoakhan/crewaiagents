"""Basic unit test for SentimentTool (mocked) and API health."""

import pytest
from fastapi.testclient import TestClient
from call_analytics.api import app
from call_analytics.tools.sentiment_tool import SentimentTool

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Call Analytics Demo" in resp.text


def test_sentiment_tool_mock(monkeypatch):
    """Mock OpenAI call so test runs offline."""

    def fake_chat_create(*_, **__):  # noqa: D401
        class Obj:
            choices = [type("Msg", (), {"message": type("M", (), {"content": '{"sentiment":"neutral","score":0}'})()})]
        return Obj()

    monkeypatch.setattr("openai.OpenAI", lambda: type("C", (), {"chat": type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(fake_chat_create)})()})()})

    tool = SentimentTool()
    out = tool.run({"text": "It was okay."})
    assert "neutral" in out
