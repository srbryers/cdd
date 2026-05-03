"""Tests for cdd.types — the core data types of CDD.

The types in this module are part of the log schema. Changing their
field names, removing values, or breaking immutability invalidates
existing logs. These tests pin the contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict


# ---------- Verdict ----------


def test_verdict_string_values_are_stable() -> None:
    """Verdict values are persisted in logs — renaming would break replay."""
    assert Verdict.ACCEPT == "accept"
    assert Verdict.REJECT == "reject"
    assert Verdict.REFINE == "refine"
    assert Verdict.ABANDON == "abandon"


def test_verdict_construct_from_string() -> None:
    assert Verdict("accept") is Verdict.ACCEPT
    assert Verdict("abandon") is Verdict.ABANDON


def test_verdict_unknown_string_raises() -> None:
    with pytest.raises(ValueError):
        Verdict("approve")  # not a real verdict


# ---------- DeterminismTier ----------


def test_determinism_tier_string_values_are_stable() -> None:
    assert DeterminismTier.STRICT_A == "strict_a"
    assert DeterminismTier.SEMANTIC_B == "semantic_b"
    assert DeterminismTier.BEST_EFFORT_C == "best_effort_c"


def test_determinism_tier_construct_from_string() -> None:
    assert DeterminismTier("semantic_b") is DeterminismTier.SEMANTIC_B


# ---------- ModelIdentity ----------


def test_model_identity_frozen() -> None:
    m = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    with pytest.raises(FrozenInstanceError):
        m.provider = "openai"  # type: ignore[misc]


def test_model_identity_hashable_and_value_equal() -> None:
    a = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    b = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_model_identity_str_unversioned() -> None:
    m = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    assert str(m) == "anthropic/claude-opus-4-7"


def test_model_identity_str_versioned() -> None:
    m = ModelIdentity(
        provider="anthropic",
        model="claude-opus-4-7",
        version="20260101",
    )
    assert str(m) == "anthropic/claude-opus-4-7@20260101"


def test_model_identity_version_default_is_none() -> None:
    m = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    assert m.version is None


def test_model_identity_inequality_via_version() -> None:
    a = ModelIdentity(provider="anthropic", model="claude-opus-4-7")
    b = ModelIdentity(
        provider="anthropic",
        model="claude-opus-4-7",
        version="20260101",
    )
    assert a != b


# ---------- LogEntry ----------


def _entry(**overrides: object) -> LogEntry:
    """Test factory — defaults for an ACCEPT entry; override any field."""
    defaults: dict[str, object] = {
        "id": "entry-1",
        "parent_id": None,
        "iteration": 1,
        "timestamp": datetime(2026, 5, 3, 12, 0, tzinfo=UTC),
        "prompt": "describe a Caribbean cove at dawn",
        "seed": 42,
        "model": ModelIdentity(provider="anthropic", model="claude-opus-4-7"),
        "extra_params": {"temperature": 0.7},
        "output_ref": "artifacts/cove-1.txt",
        "verdict": Verdict.ACCEPT,
        "notes": "warm light works",
    }
    defaults.update(overrides)
    return LogEntry(**defaults)  # type: ignore[arg-type]


def test_log_entry_frozen() -> None:
    e = _entry()
    with pytest.raises(FrozenInstanceError):
        e.prompt = "different"  # type: ignore[misc]


def test_log_entry_first_iteration_has_no_parent() -> None:
    e = _entry(parent_id=None, iteration=1)
    assert e.parent_id is None


def test_log_entry_chain_via_parent_id() -> None:
    first = _entry(id="entry-1", parent_id=None, iteration=1, verdict=Verdict.REJECT)
    second = _entry(
        id="entry-2", parent_id="entry-1", iteration=2, verdict=Verdict.ACCEPT
    )
    assert second.parent_id == first.id


def test_log_entry_seed_may_be_none() -> None:
    """LLM providers often don't honor seeds — None must be allowed."""
    e = _entry(seed=None)
    assert e.seed is None


def test_log_entry_default_notes_is_empty_string() -> None:
    e = LogEntry(
        id="x",
        parent_id=None,
        iteration=1,
        timestamp=datetime(2026, 5, 3, tzinfo=UTC),
        prompt="x",
        seed=None,
        model=ModelIdentity(provider="x", model="y"),
        extra_params={},
        output_ref="x",
        verdict=Verdict.ACCEPT,
    )
    assert e.notes == ""


def test_log_entry_value_equality() -> None:
    a = _entry()
    b = _entry()
    assert a == b
