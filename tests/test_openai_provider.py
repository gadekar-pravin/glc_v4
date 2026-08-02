"""OpenAI as a provider in its own right, rather than a vendor prefix on someone
else's model name.

It was added because GitHub Models began returning 410 and took the only
frontier-rate model this gateway could reach with it. Three things about it are
unlike the five servers that already speak the OpenAI dialect, and each one fails
quietly rather than loudly:

* the gpt-5 family rejects `max_tokens` and wants `max_completion_tokens`, and
  the 400-healing loop only knows how to strip *reasoning* keys, so the wrong
  name is a hard 400;
* a provider registered with no `LIMITS` row raises `KeyError` on the first pick,
  not on startup;
* an OpenAI model that reaches no pricing row would bill $0.00 and walk straight
  through the budget controller.
"""

from __future__ import annotations

import pytest

from glc import providers as P
from glc.economics import pricing as PR
from glc.routing import LIMITS, Router, resolve


@pytest.fixture(autouse=True)
def _fresh_table():
    PR.reload_pricing()
    yield
    PR.reload_pricing()


def _pool(monkeypatch, **env):
    """Build the worker pool from a known environment.

    Every OpenAI variable is cleared first. `glc/main.py` calls `load_dotenv()` at
    import, so on the machine that actually has a key these tests would otherwise
    assert against the developer's `.env` and pass or fail for reasons that never
    reproduce in CI.
    """
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return P.build_providers(cache_store=object())


# ── the output ceiling ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_asks_for_max_completion_tokens(monkeypatch):
    """`max_tokens` is what every other server on this surface reads, and the one
    thing OpenAI's own will not accept."""
    seen: list[dict] = []

    class _R:
        status_code, text = 200, ""

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            seen.append(dict(json))
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    provider = P.OpenAIProvider("k", "gpt-5.6-terra")
    out = await provider.chat([{"role": "user", "content": "hi"}], max_tokens=4096)

    assert out["text"] == "ok"
    assert seen[-1]["max_completion_tokens"] == 4096
    assert "max_tokens" not in seen[-1]
    assert provider.base_url == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_a_fixed_temperature_model_is_sent_no_temperature_at_all(monkeypatch):
    """MEASURED: gpt-5.6-terra answers `temperature: 0` with HTTP 400
    `unsupported_value`. Nothing heals it — the retry loop only strips reasoning
    keys — so every frontier call 400s until the field is omitted."""
    seen: list[dict] = []

    class _R:
        status_code, text = 200, ""

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            seen.append(dict(json))
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())

    out = await P.OpenAIProvider("k", "gpt-5.6-terra").chat(
        [{"role": "user", "content": "hi"}], max_tokens=512, temperature=0
    )
    assert "temperature" not in seen[-1]
    assert out["temperature_applied"] is False

    # Only the family that actually rejects it. Every other model keeps the dial,
    # because omitting it silently costs determinism.
    out = await P.OpenAIProvider("k", "gpt-4o").chat(
        [{"role": "user", "content": "hi"}], max_tokens=512, temperature=0
    )
    assert seen[-1]["temperature"] == 0
    assert out["temperature_applied"] is True


@pytest.mark.asyncio
async def test_every_other_provider_still_sends_max_tokens(monkeypatch):
    """The field name is a class attribute on the shared base, so this is the
    regression that would follow a careless rename."""
    seen: list[dict] = []

    class _R:
        status_code, text = 200, ""

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            seen.append(dict(json))
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    for provider in (
        P.GroqProvider("k", "openai/gpt-oss-120b"),
        P.CerebrasProvider("k", "zai-glm-4.7"),
        P.NvidiaProvider("k", "deepseek-ai/deepseek-v4-pro"),
        P.OpenRouterProvider("k", "nvidia/nemotron-3-super-120b-a12b:free"),
        P.GitHubProvider("k", "openai/gpt-4.1"),
    ):
        await provider.chat([{"role": "user", "content": "hi"}], max_tokens=64)
        assert seen[-1]["max_tokens"] == 64, provider.name
        assert "max_completion_tokens" not in seen[-1], provider.name


# ── registration ────────────────────────────────────────────────────────────


def test_the_key_registers_openai_and_its_absence_omits_it(monkeypatch):
    assert "openai" not in _pool(monkeypatch)

    pool = _pool(monkeypatch, OPENAI_API_KEY="k")
    assert isinstance(pool["openai"], P.OpenAIProvider)
    assert pool["openai"].model == "gpt-5.6-terra"

    pinned = _pool(monkeypatch, OPENAI_API_KEY="k", OPENAI_MODEL="gpt-5.6-luna")
    assert pinned["openai"].model == "gpt-5.6-luna"


def test_a_registered_openai_can_actually_be_picked(monkeypatch):
    """`LIMITS` is a plain dict indexed with `LIMITS[name]`, so a provider with no
    row is a KeyError at pick time — long after the gateway reported healthy."""
    assert "openai" in LIMITS
    pool = _pool(monkeypatch, OPENAI_API_KEY="k")
    router = Router(pool, ["openai"])
    assert router.candidates() == ["openai"]
    picked, _ = router.pick(100, router.candidates())
    assert picked == "openai"
    assert router.all_status()["openai"]


def test_the_shortcut_resolves():
    assert resolve("oa") == resolve("oai") == resolve("openai") == "openai"


def test_reasoning_is_resolved_per_model_not_left_at_the_class_default(monkeypatch):
    """The class says True; only `model_capabilities` knows whether *this* model
    thinks. Without "openai" in its provider tuple the flag is never checked, and
    `Router.pick(required_caps=["reasoning"])` trusts it."""
    pool = _pool(monkeypatch, OPENAI_API_KEY="k", OPENAI_MODEL="gpt-5.6-terra")
    assert pool["openai"].capabilities["reasoning"] is True
    assert P.model_capabilities("openai", "text-embedding-3-large", {"reasoning": True})["reasoning"] is False


# ── pricing ─────────────────────────────────────────────────────────────────


def test_the_frontier_model_bills_its_list_rate():
    p = PR.price_for("openai", "gpt-5.6-terra")
    assert p.source == "model"
    assert (p.input_usd_per_mtok, p.output_usd_per_mtok) == (2.00, 12.00)


def test_luna_carries_the_post_cut_rate_not_the_launch_rate():
    """1.00/6.00 was the launch price and is still quoted widely; OpenAI cut Luna
    80% on 2026-07-30. OpenRouter's 0.10/0.60 is a reseller promo, not list."""
    p = PR.price_for("openai", "gpt-5.6-luna")
    assert (p.input_usd_per_mtok, p.output_usd_per_mtok) == (0.20, 1.20)


def test_no_openai_model_is_ever_free():
    """The failure this guards is silent: an unpriced model costs $0.00, so the
    budget controller admits it forever."""
    for model in ("gpt-5.6-sol", "gpt-5.7-whatever-ships-next", "o5-mini", ""):
        p = PR.price_for("openai", model)
        assert p.input_usd_per_mtok > 0 and p.output_usd_per_mtok > 0, (model, p.source)


def test_the_gpt5_glob_does_not_swallow_cheaper_gpt_models():
    """The glob is `gpt-5*` rather than `gpt-*` for these two.

    A bare `gpt-oss-120b` has no row of its own and would have been caught by the
    wider glob and billed 13x its real rate. `openai/gpt-4.1-mini` is safe by two
    separate mechanisms — it has an exact row, and globs match the full model
    string — but only the first of those survives someone adding a suffix match.
    """
    oss = PR.price_for("groq", "gpt-oss-120b")
    assert oss.source == "provider"
    assert (oss.input_usd_per_mtok, oss.output_usd_per_mtok) == (0.15, 0.75)

    mini = PR.price_for("github", "openai/gpt-4.1-mini")
    assert mini.source == "model"
    assert (mini.input_usd_per_mtok, mini.output_usd_per_mtok) == (0.40, 1.60)


def test_openai_is_last_in_the_default_order():
    """It is the dearest thing this gateway can reach, so it must be the last
    resort of the fallback ring rather than an early candidate.

    Only `DEFAULT_ORDER` is asserted. Whether the running gateway actually offers
    OpenAI depends on `LLM_ORDER` in a `.env` this repo does not track, and a test
    that read it would pass or fail per developer and never in CI.
    """
    from glc.routes import chat as C

    assert C.DEFAULT_ORDER[-1] == "openai"
