"""The Groq provider: what it sends, what it streams, and how it fails.

No network: the SDK client is replaced by a fake that records each request
and plays back chunks shaped like Groq's streaming response.
"""

from types import SimpleNamespace

import groq
import httpx
import pytest

from backend import llm
from backend.providers import groq_provider
from backend.providers.base import ProviderError, ProviderNotReady
from backend.providers.groq_provider import GroqProvider
from utils import config

pytestmark = pytest.mark.roadmap("0.6")


def chunk(content=None, reasoning=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage, x_groq=None)


def usage(prompt=10, completion=30, reasoning=20):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def status_error(cls, code, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    response = httpx.Response(code, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    return cls(f"error {code}", response=response, body=None)


class FakeGroq:
    """Answers each create() with the next scripted outcome, per model."""

    def __init__(self, script):
        self.script = script          # model -> list of (chunks | exception)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.script[kwargs["model"]].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return iter(outcome)


@pytest.fixture
def fake(monkeypatch):
    slept = []
    monkeypatch.setattr(groq_provider.time, "sleep", slept.append)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-env")

    def install(script):
        client = FakeGroq(script)
        client.slept = slept
        monkeypatch.setattr(GroqProvider, "_client", lambda self: client)
        return client
    return install


STRONG, CHEAP = config.GROQ_MODEL, config.GROQ_CHEAP_MODEL


def test_streams_answer_text_and_never_the_reasoning(fake):
    fake({STRONG: [[
        chunk(reasoning="Let me think about revenue..."),
        chunk(content="Revenue was "), chunk(content="$391bn [1]."),
        chunk(usage=usage(prompt=100, completion=60, reasoning=40)),
    ]]})
    provider = GroqProvider()
    assert "".join(provider.stream("sys", "q")) == "Revenue was $391bn [1]."
    assert provider.last_usage == {
        "model": STRONG, "prompt_tokens": 100, "completion_tokens": 60, "reasoning_tokens": 40,
    }


def test_sends_effort_temperature_and_hides_reasoning(fake):
    client = fake({STRONG: [[chunk(content="ok")]]})
    list(GroqProvider(reasoning_effort="high", temperature=0.3).stream("sys", "q"))
    sent = client.requests[0]
    assert sent["model"] == STRONG
    assert sent["reasoning_effort"] == "high"
    assert sent["temperature"] == 0.3
    assert sent["include_reasoning"] is False
    assert sent["stream"] is True
    assert sent["messages"][0] == {"role": "system", "content": "sys"}


def test_each_stage_gets_its_model_and_effort():
    provider = GroqProvider()
    for stage, (model, effort) in config.GROQ_STAGES.items():
        staged = provider.for_stage(stage)
        assert (staged.model, staged.reasoning_effort) == (model, effort)
    assert provider.for_stage("classify").model == CHEAP
    assert provider.for_stage("answer").model == STRONG
    assert provider.for_stage("not-a-stage").model == config.GROQ_MODEL
    assert (provider.model, provider.reasoning_effort) == (config.GROQ_MODEL, config.GROQ_REASONING_EFFORT), \
        "for_stage changed the original"


def test_a_staged_copy_keeps_the_typed_key():
    assert GroqProvider(api_key="typed").for_stage("classify").api_key == "typed"


def test_rate_limited_model_falls_through_to_the_next(fake):
    client = fake({
        STRONG: [status_error(groq.RateLimitError, 429, retry_after=7)],
        CHEAP: [[chunk(content="from the cheap model")]],
    })
    assert "".join(GroqProvider().stream("s", "q")) == "from the cheap model"
    assert [r["model"] for r in client.requests] == [STRONG, CHEAP]
    assert client.slept == [], "waited although another model was free"


def test_waits_as_long_as_asked_then_gives_up_with_a_reason(fake):
    limited = lambda: status_error(groq.RateLimitError, 429, retry_after=7)
    client = fake({STRONG: [limited(), limited()], CHEAP: [limited(), limited()]})
    with pytest.raises(ProviderError, match="rate limit"):
        list(GroqProvider().stream("s", "q"))
    assert client.slept == [7.0]


def test_the_wait_is_capped(fake):
    limited = lambda: status_error(groq.RateLimitError, 429, retry_after=3600)
    client = fake({STRONG: [limited(), limited()], CHEAP: [limited(), limited()]})
    with pytest.raises(ProviderError):
        list(GroqProvider().stream("s", "q"))
    assert client.slept == [config.GROQ_MAX_WAIT]


def test_a_rejected_key_is_not_retried(fake):
    client = fake({STRONG: [status_error(groq.AuthenticationError, 401)]})
    with pytest.raises(ProviderError, match="rejected the API key"):
        list(GroqProvider().stream("s", "q"))
    assert len(client.requests) == 1


def test_an_unknown_model_says_where_to_look(fake):
    fake({STRONG: [status_error(groq.NotFoundError, 404)]})
    with pytest.raises(ProviderError, match="GROQ_MODEL"):
        list(GroqProvider().stream("s", "q"))


def test_an_answer_lost_to_thinking_is_explained(fake):
    fake({STRONG: [[chunk(reasoning="thinking..."), chunk(usage=usage())]]})
    with pytest.raises(ProviderError, match="GROQ_MAX_TOKENS"):
        list(GroqProvider().stream("s", "q"))


def test_no_key_is_not_ready(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderNotReady, match="console.groq.com"):
        GroqProvider().check_ready()


def test_a_typed_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert GroqProvider(api_key=" typed ")._api_key() == "typed"


def test_groq_is_the_default_provider():
    assert config.PROVIDER == "groq"
    assert isinstance(llm.get_provider(), GroqProvider)
    assert llm.model_for("groq") == config.GROQ_MODEL


def test_the_setup_checker_reports_groq_ready(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    from utils import check

    ready = check.report_readiness()
    assert "groq" in ready
    assert "-> groq" in capsys.readouterr().out
