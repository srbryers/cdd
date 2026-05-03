"""Tests for cdd.loop — the critique-session orchestrator.

These tests use minimal in-test stub adapters and generators. The
real reference adapter (TextFileAdapter) and reference generator
(EchoGenerator) are tested elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from cdd.log import Log
from cdd.loop import Loop, LoopStateError
from cdd.types import DeterminismTier, ModelIdentity, Verdict


# ---------- test stubs ----------


class _MemAdapter:
    artifact_type = "text"
    determinism_tier = DeterminismTier.SEMANTIC_B

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.persisted: list[tuple[str, str]] = []  # (namespace, raw)

    def persist(self, raw_output: Any, *, namespace: str) -> str:
        if not isinstance(raw_output, str):
            raise TypeError(f"text adapter expects str, got {type(raw_output).__name__}")
        ref = f"{namespace}/{len(self.store)}"
        self.store[ref] = raw_output
        self.persisted.append((namespace, raw_output))
        return ref

    def load(self, output_ref: str) -> Any:
        return self.store[output_ref]


class _StrictAdapter(_MemAdapter):
    determinism_tier = DeterminismTier.STRICT_A


class _EchoGen:
    supports_seed = True

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider="stub", model="echo", version="v1")

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        return f"echo[seed={seed}]: {prompt}"


# ---------- construction ----------


def test_loop_creates_log_with_session_id() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen(), session_id="loop-42")
    assert loop.log.session_id == "loop-42"


def test_loop_auto_generates_session_id_uuid() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    assert loop.log.session_id  # non-empty
    # UUID4 has 4 hyphens
    assert loop.log.session_id.count("-") == 4


def test_loop_log_inherits_artifact_type_and_tier_from_adapter() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    assert loop.log.artifact_type == "text"
    assert loop.log.determinism_tier is DeterminismTier.SEMANTIC_B


def test_loop_starts_with_no_pending_no_resolved() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    assert loop.is_pending is False
    assert loop.is_resolved is False
    assert loop.current_output() is None


# ---------- propose ----------


def test_propose_returns_raw_output() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    out = loop.propose("describe a Caribbean cove")
    assert out == "echo[seed=None]: describe a Caribbean cove"


def test_propose_persists_via_adapter() -> None:
    adapter = _MemAdapter()
    loop = Loop(adapter=adapter, generator=_EchoGen(), session_id="sess-x")
    loop.propose("Nassau dawn")
    assert len(adapter.persisted) == 1
    namespace, raw = adapter.persisted[0]
    assert namespace == "sess-x"
    assert "Nassau dawn" in raw


def test_propose_marks_pending() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("anything")
    assert loop.is_pending is True
    assert loop.current_output() is not None


def test_propose_twice_without_review_raises() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("first")
    with pytest.raises(LoopStateError, match="pending review"):
        loop.propose("second")


def test_propose_passes_seed_to_generator() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    out = loop.propose("anything", seed=99)
    assert "seed=99" in out


# ---------- review ----------


def test_review_without_pending_raises() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    with pytest.raises(LoopStateError, match="no pending proposal"):
        loop.review(Verdict.ACCEPT)


def test_review_appends_log_entry() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    entry = loop.review(Verdict.REJECT, notes="too literal")
    assert len(loop.log.entries) == 1
    assert entry.verdict is Verdict.REJECT
    assert entry.notes == "too literal"
    assert entry.iteration == 1
    assert entry.parent_id is None


def test_review_clears_pending() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    loop.review(Verdict.REJECT)
    assert loop.is_pending is False
    assert loop.current_output() is None


def test_review_chains_parent_id() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("first")
    e1 = loop.review(Verdict.REJECT)
    loop.propose("second")
    e2 = loop.review(Verdict.ACCEPT)
    assert e2.parent_id == e1.id
    assert e2.iteration == 2


def test_review_records_model_identity_at_propose_time() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("anything", seed=7)
    entry = loop.review(Verdict.ACCEPT)
    assert entry.model.provider == "stub"
    assert entry.model.model == "echo"
    assert entry.model.version == "v1"
    assert entry.seed == 7


def test_review_records_extra_params() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("anything", extra_params={"temperature": 0.3})
    entry = loop.review(Verdict.ACCEPT)
    assert entry.extra_params == {"temperature": 0.3}


# ---------- resolution ----------


def test_review_accept_resolves_loop() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    loop.review(Verdict.ACCEPT)
    assert loop.is_resolved is True


def test_review_abandon_resolves_loop() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    loop.review(Verdict.ABANDON)
    assert loop.is_resolved is True


def test_propose_after_accept_raises() -> None:
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    loop.review(Verdict.ACCEPT)
    with pytest.raises(LoopStateError, match="resolved"):
        loop.propose("again")


def test_propose_after_reject_works() -> None:
    """REJECT does not resolve the loop — it's a "try again" verdict."""
    loop = Loop(adapter=_MemAdapter(), generator=_EchoGen())
    loop.propose("once")
    loop.review(Verdict.REJECT)
    loop.propose("again")
    assert loop.is_pending is True


# ---------- resume from existing log ----------


def test_loop_resumes_from_unresolved_log() -> None:
    log = Log(
        session_id="resumed",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    loop1 = Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)
    loop1.propose("only iteration")
    loop1.review(Verdict.REJECT)

    # Resume in a new Loop using the same log
    loop2 = Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)
    assert loop2.log.session_id == "resumed"
    assert len(loop2.log.entries) == 1
    loop2.propose("retry")
    e = loop2.review(Verdict.ACCEPT)
    assert e.iteration == 2
    assert e.parent_id == log.entries[0].id


def test_resume_from_resolved_log_raises() -> None:
    log = Log(
        session_id="done",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    loop1 = Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)
    loop1.propose("once")
    loop1.review(Verdict.ACCEPT)
    with pytest.raises(LoopStateError, match="resolved"):
        Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)


def test_resume_with_mismatched_artifact_type_raises() -> None:
    log = Log(
        session_id="x",
        artifact_type="vegetation_density",  # mismatch
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    with pytest.raises(LoopStateError, match="artifact_type"):
        Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)


def test_resume_with_mismatched_tier_raises() -> None:
    log = Log(
        session_id="x",
        artifact_type="text",
        determinism_tier=DeterminismTier.STRICT_A,  # mismatch with _MemAdapter (B)
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    with pytest.raises(LoopStateError, match="determinism_tier"):
        Loop(adapter=_MemAdapter(), generator=_EchoGen(), log=log)


def test_resume_with_disagreeing_session_id_raises() -> None:
    """Passing both ``session_id`` and ``log`` with different session IDs is a footgun."""
    log = Log(
        session_id="logged-session",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    with pytest.raises(LoopStateError, match="session_id"):
        Loop(
            adapter=_MemAdapter(),
            generator=_EchoGen(),
            session_id="conflicting-id",
            log=log,
        )


def test_resume_with_matching_session_id_is_allowed() -> None:
    """If the explicit session_id matches the log's, it's redundant but fine."""
    log = Log(
        session_id="my-session",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    loop = Loop(
        adapter=_MemAdapter(),
        generator=_EchoGen(),
        session_id="my-session",
        log=log,
    )
    assert loop.log.session_id == "my-session"
