"""Semantic cache: cosine matching, the threshold, namespace isolation, and
the measured tokens/dollars a hit saves."""

from __future__ import annotations

import math

import pytest

from glc import db
from glc.cache import GeminiCache, cosine
from glc.cache.semantic import SemanticCache, SemanticCacheConfig


@pytest.fixture(autouse=True)
def _fresh():
    db.init()
    yield


def _fixed_embedder(mapping: dict[str, list[float]], default=None):
    """Deterministic embedder: text -> vector, no network."""

    async def embed(text, task_type):  # noqa: ARG001
        if text in mapping:
            return mapping[text]
        return default if default is not None else [1.0, 0.0, 0.0]

    return embed


def _cache(**overrides) -> SemanticCache:
    kw = {
        "enabled": True,
        "default_on": True,
        "threshold": 0.95,
        "ttl_seconds": 0,
        "max_entries": 0,
        "scope_dimensions": [],
    }
    kw.update(overrides)
    return SemanticCache(config=SemanticCacheConfig(**kw), embed_fn=_fixed_embedder({}))


# ── the package shim ────────────────────────────────────────────────────────


def test_gemini_cache_import_path_is_unchanged():
    """`from glc.cache import GeminiCache` was the v3 import; it must survive
    cache.py becoming a package."""
    c = GeminiCache(ttl_seconds=300)
    assert c.ttl == 300
    assert GeminiCache._key("m", "text") == GeminiCache._key("m", "text")
    assert GeminiCache._key("m", "a") != GeminiCache._key("m", "b")


# ── cosine ──────────────────────────────────────────────────────────────────


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    # magnitude-invariant
    assert cosine([3, 4], [6, 8]) == pytest.approx(1.0)


def test_cosine_is_defensive_about_bad_input():
    assert cosine([], [1]) == 0.0
    assert cosine([1, 2], [1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0


# ── hit / miss ──────────────────────────────────────────────────────────────


async def test_a_hit_returns_the_stored_response_and_counts_the_saving():
    v = [1.0, 0.0, 0.0]
    c = _cache()
    c.embed_fn = _fixed_embedder({"what is 2+2": v, "what is two plus two": v})
    fields = {"model": "m", "system": None}

    miss = await c.lookup("what is 2+2", fields)
    assert miss.hit is False

    await c.store(
        "what is 2+2",
        fields,
        response={"text": "4", "provider": "gemini_1"},
        provider="gemini_1",
        model="gemini-3.1-flash-lite",
        input_tokens=1200,
        output_tokens=40,
        usd=0.00036,
    )

    hit = await c.lookup("what is two plus two", fields)
    assert hit.hit is True
    assert hit.similarity == pytest.approx(1.0)
    assert hit.response["text"] == "4"
    assert hit.tokens_saved == 1240
    assert hit.usd_saved == pytest.approx(0.00036)


async def test_below_threshold_is_a_miss_and_records_how_close_it_got():
    theta = math.radians(30)  # cos 30 deg = 0.866, below 0.95
    c = _cache()
    c.embed_fn = _fixed_embedder({"a": [1.0, 0.0], "b": [math.cos(theta), math.sin(theta)]})
    fields = {"model": "m"}
    await c.store("a", fields, {"text": "A"}, provider="p", model="m", input_tokens=10, output_tokens=1)
    lookup = await c.lookup("b", fields)
    assert lookup.hit is False
    assert lookup.best_similarity == pytest.approx(0.866, abs=1e-3)
    assert c.stats()["best_similarity_missed"] == pytest.approx(0.866, abs=1e-3)


async def test_threshold_is_configurable_and_changes_the_verdict():
    theta = math.radians(30)
    emb = _fixed_embedder({"a": [1.0, 0.0], "b": [math.cos(theta), math.sin(theta)]})
    fields = {"model": "m"}

    strict = _cache(threshold=0.95)
    strict.embed_fn = emb
    await strict.store("a", fields, {"text": "A"}, provider="p", model="m")
    assert (await strict.lookup("b", fields)).hit is False

    loose = _cache(threshold=0.80)
    loose.embed_fn = emb
    assert (await loose.lookup("b", fields)).hit is True


# ── isolation ───────────────────────────────────────────────────────────────


async def test_a_different_model_is_a_different_namespace():
    """Identical text, different model: a cached answer must not cross over."""
    v = [1.0, 0.0]
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": v})
    await c.store("q", {"model": "cheap"}, {"text": "A"}, provider="p", model="cheap")
    assert (await c.lookup("q", {"model": "cheap"})).hit is True
    assert (await c.lookup("q", {"model": "expensive"})).hit is False


async def test_a_different_system_prompt_is_a_different_namespace():
    v = [1.0, 0.0]
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": v})
    await c.store("q", {"model": "m", "system": "be terse"}, {"text": "A"}, provider="p", model="m")
    assert (await c.lookup("q", {"model": "m", "system": "be terse"})).hit is True
    assert (await c.lookup("q", {"model": "m", "system": "be verbose"})).hit is False


async def test_scope_dimensions_stop_cross_tenant_leakage():
    from glc.economics.meter import Principal

    v = [1.0, 0.0]
    c = _cache(scope_dimensions=["tenant"])
    c.embed_fn = _fixed_embedder({"q": v})
    fields = {"model": "m"}
    await c.store(
        "q", fields, {"text": "acme secret"}, provider="p", model="m", principal=Principal(tenant="acme")
    )
    assert (await c.lookup("q", fields, principal=Principal(tenant="acme"))).hit is True
    assert (await c.lookup("q", fields, principal=Principal(tenant="other"))).hit is False


# ── admission rules ─────────────────────────────────────────────────────────


def test_ships_opt_in_not_on():
    """A gateway that silently starts replaying answers is a surprise."""
    cfg = SemanticCacheConfig.load()
    assert cfg.enabled is True
    assert cfg.default_on is False
    assert cfg.threshold >= 0.92  # the safe production floor


def test_should_consult_rules():
    c = _cache()
    assert c.should_consult(opt_in=True)[0] is True
    assert c.should_consult(opt_in=False)[0] is False
    # high temperature means the caller asked for variety
    ok, why = c.should_consult(opt_in=True, temperature=0.9)
    assert ok is False and "variety" in why
    # tool calls must execute against current state
    ok, why = c.should_consult(opt_in=True, has_tools=True)
    assert ok is False and "tool definitions" in why
    ok, why = c.should_consult(opt_in=True, stream=True)
    assert ok is False and "streaming" in why


def test_disabled_cache_never_consults():
    c = _cache()
    c.config.enabled = False
    ok, why = c.should_consult(opt_in=True)
    assert ok is False and "disabled" in why


async def test_default_on_makes_it_implicit():
    c = _cache(default_on=True)
    assert c.should_consult(opt_in=None)[0] is True
    c2 = _cache()
    c2.config.default_on = False
    assert c2.should_consult(opt_in=None)[0] is False


async def test_no_embedder_is_a_graceful_skip_not_a_failure():
    c = SemanticCache(
        config=SemanticCacheConfig(enabled=True, default_on=True, scope_dimensions=[]), embed_fn=None
    )
    lookup = await c.lookup("q", {"model": "m"})
    assert lookup.hit is False
    assert "no embedding available" in lookup.skipped_reason


async def test_an_embedder_that_raises_is_counted_and_survived():
    async def boom(text, task_type):  # noqa: ARG001
        raise RuntimeError("embedder down")

    c = _cache()
    c.embed_fn = boom
    lookup = await c.lookup("q", {"model": "m"})
    assert lookup.hit is False
    assert c.stats()["embed_failures"] == 1


# ── ttl, eviction, stats ────────────────────────────────────────────────────


async def test_expired_entries_are_not_served():
    v = [1.0, 0.0]
    c = _cache(ttl_seconds=1)
    c.embed_fn = _fixed_embedder({"q": v})
    fields = {"model": "m"}
    await c.store("q", fields, {"text": "A"}, provider="p", model="m")
    assert (await c.lookup("q", fields)).hit is True
    # age the row past the TTL
    with db.conn() as conn:
        conn.execute("UPDATE semantic_cache SET created_at = created_at - 10")
    assert (await c.lookup("q", fields)).hit is False
    assert c.purge(expired_only=True) == 1


async def test_max_entries_evicts_the_oldest():
    c = _cache(max_entries=2)
    c.embed_fn = _fixed_embedder({}, default=[1.0, 0.0])
    for i in range(5):
        await c.store(f"q{i}", {"model": "m"}, {"text": str(i)}, provider="p", model="m")
    assert c.stats()["entries"] == 2


async def test_stats_report_hit_rate_and_totals():
    v = [1.0, 0.0]
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": v})
    fields = {"model": "m"}
    await c.lookup("q", fields)  # miss
    await c.store(
        "q", fields, {"text": "A"}, provider="p", model="m", input_tokens=100, output_tokens=10, usd=0.5
    )
    await c.lookup("q", fields)  # hit
    await c.lookup("q", fields)  # hit
    s = c.stats()
    assert s["lookups"] == 3
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["tokens_saved"] == 220
    assert s["usd_saved"] == pytest.approx(1.0)
    assert s["entries"] == 1
    assert s["config"]["threshold"] == 0.95


async def test_hits_persist_across_a_new_cache_object():
    """Storage is the gateway's SQLite file, so the hit rate is real history."""
    v = [1.0, 0.0]
    c1 = _cache()
    c1.embed_fn = _fixed_embedder({"q": v})
    await c1.store("q", {"model": "m"}, {"text": "A"}, provider="p", model="m")

    c2 = _cache()
    c2.embed_fn = _fixed_embedder({"q": v})
    assert (await c2.lookup("q", {"model": "m"})).hit is True


def test_config_comes_from_yaml_with_no_python_edit(tmp_path, monkeypatch):
    y = tmp_path / "cache.yaml"
    y.write_text(
        "version: 1\nsemantic:\n  enabled: true\n  default_on: true\n  threshold: 0.111\n"
        "  ttl_seconds: 42\n  scope_dimensions: [project]\n"
    )
    monkeypatch.setenv("GLC_CACHE_YAML", str(y))
    cfg = SemanticCacheConfig.load()
    assert cfg.threshold == 0.111
    assert cfg.ttl_seconds == 42
    assert cfg.default_on is True
    assert cfg.scope_dimensions == ["project"]


# ── what is worth storing ───────────────────────────────────────────────────
#
# A hit skips the provider entirely, so whatever is in the entry is the final
# answer for the whole TTL. An answer the model never finished saying is not
# an answer, and caching one turns a single truncated generation into an hour
# of confidently truncated replies.


async def test_a_truncated_response_is_not_stored():
    """stop_reason max_tokens means the model was cut off mid-sentence."""
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": [1.0, 0.0, 0.0]})
    fields = {"model": "m"}

    row_id = await c.store(
        "q",
        fields,
        response={"text": "The tallest mountain on Earth, measured", "stop_reason": "max_tokens"},
        provider="p",
        model="m",
        output_tokens=7,
    )

    assert row_id is None
    assert (await c.lookup("q", fields)).hit is False
    assert c.stats()["stores"] == 0


async def test_an_empty_response_is_never_stored_even_if_it_finished():
    """A thinking model can spend its whole budget reasoning and emit no text.

    Unlike truncation this is not a policy question — an empty cached answer
    has no use at any setting, so it is refused even with the knob off.
    """
    c = _cache(skip_when_truncated=False)
    c.embed_fn = _fixed_embedder({"q": [1.0, 0.0, 0.0]})
    fields = {"model": "m"}

    row_id = await c.store(
        "q", fields, response={"text": "", "stop_reason": "end_turn"}, provider="p", model="m"
    )

    assert row_id is None
    assert (await c.lookup("q", fields)).hit is False


async def test_storing_truncated_responses_is_configurable():
    """The knob is YAML, not a Python edit — same as every other cache rule."""
    c = _cache(skip_when_truncated=False)
    c.embed_fn = _fixed_embedder({"q": [1.0, 0.0, 0.0]})
    fields = {"model": "m"}

    row_id = await c.store(
        "q", fields, response={"text": "cut off here", "stop_reason": "max_tokens"}, provider="p", model="m"
    )

    assert row_id is not None
    assert (await c.lookup("q", fields)).hit is True


async def test_a_complete_response_is_still_stored():
    """The guard must not cost us the ordinary case."""
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": [1.0, 0.0, 0.0]})
    fields = {"model": "m"}

    row_id = await c.store(
        "q", fields, response={"text": "Mount Everest.", "stop_reason": "end_turn"}, provider="p", model="m"
    )

    assert row_id is not None
    assert (await c.lookup("q", fields)).hit is True
    assert c.stats()["stores"] == 1


def test_should_store_rules():
    """Symmetric with should_consult: a verdict plus a human-readable reason."""
    c = _cache()

    ok, reason = c.should_store({"text": "a complete answer", "stop_reason": "end_turn"})
    assert ok is True
    assert reason == ""

    ok, reason = c.should_store({"text": "cut off", "stop_reason": "max_tokens"})
    assert ok is False
    assert "truncat" in reason.lower()

    ok, reason = c.should_store({"text": "   ", "stop_reason": "end_turn"})
    assert ok is False
    assert "empty" in reason.lower()


def test_truncation_guard_ships_on():
    assert SemanticCacheConfig().skip_when_truncated is True


async def test_an_errored_generation_is_not_stored():
    """`error` is the other stop_reason in the envelope that means the model
    did not finish cleanly (glc/llm_schemas.py). Partial text from a failed
    generation is not an answer either."""
    c = _cache()
    c.embed_fn = _fixed_embedder({"q": [1.0, 0.0, 0.0]})
    fields = {"model": "m"}

    row_id = await c.store(
        "q", fields, response={"text": "partial output", "stop_reason": "error"}, provider="p", model="m"
    )

    assert row_id is None
    assert (await c.lookup("q", fields)).hit is False
