"""End-to-end integration tests for CDD.

Exercises the full critique cycle using the reference adapter
(TextFileAdapter on real filesystem) and the reference generator
(EchoGenerator). These tests demonstrate the public API surface a
real CDD user would touch:

    Loop -> propose -> review -> log -> write_log -> read_log -> resume
"""

from __future__ import annotations

from pathlib import Path

from cdd.adapters.text import TextFileAdapter
from cdd.generators.echo import EchoGenerator
from cdd.log import read_log, write_log
from cdd.loop import Loop
from cdd.types import Verdict


def test_end_to_end_critique_cycle(tmp_path: Path) -> None:
    """A full session: 2 rejections, 1 refinement, 1 acceptance."""
    adapter = TextFileAdapter(tmp_path / "artifacts")
    generator = EchoGenerator()

    loop = Loop(adapter=adapter, generator=generator, session_id="integration-1")

    # iteration 1 — reject
    loop.propose("describe a cove")
    loop.review(Verdict.REJECT, notes="too brief")

    # iteration 2 — reject
    loop.propose("describe a cove")
    loop.review(Verdict.REJECT, notes="needs more detail")

    # iteration 3 — refine (new prompt)
    loop.propose("describe a cove with palm trees and limestone")
    loop.review(Verdict.REFINE, notes="closer; want dawn light")

    # iteration 4 — accept
    accepted_raw = loop.propose(
        "describe a cove with palm trees, limestone, and warm dawn light"
    )
    accepted_entry = loop.review(Verdict.ACCEPT, notes="exactly right")

    # session is resolved
    assert loop.is_resolved is True
    assert len(loop.log.entries) == 4
    assert loop.log.accepted_entry() is not None
    assert loop.log.accepted_entry() == accepted_entry  # type: ignore[comparison-overlap]

    # accepted output reloads from disk via the adapter
    reloaded = adapter.load(accepted_entry.output_ref)
    assert reloaded == accepted_raw

    # the artifact file actually exists where the adapter said
    artifact_path = (tmp_path / "artifacts") / accepted_entry.output_ref
    assert artifact_path.exists()


def test_session_persists_and_resumes(tmp_path: Path) -> None:
    """Write a partial session, reload it, finish it."""
    adapter = TextFileAdapter(tmp_path / "artifacts")
    generator = EchoGenerator()
    log_path = tmp_path / "session.cdd.yaml"

    # Phase 1 — start a session with one rejected proposal
    loop1 = Loop(adapter=adapter, generator=generator, session_id="resumable")
    loop1.propose("first attempt")
    loop1.review(Verdict.REJECT, notes="not yet")
    write_log(loop1.log, log_path)

    # Phase 2 — resume from the saved log, finish with accept
    restored_log = read_log(log_path)
    assert restored_log.session_id == "resumable"
    assert len(restored_log.entries) == 1

    loop2 = Loop(adapter=adapter, generator=generator, log=restored_log)
    loop2.propose("second attempt")
    final_entry = loop2.review(Verdict.ACCEPT, notes="done")

    assert final_entry.iteration == 2
    assert final_entry.parent_id == restored_log.entries[0].id
    assert loop2.is_resolved


def test_log_yaml_is_human_readable_after_full_cycle(tmp_path: Path) -> None:
    """The resulting YAML should be inspectable by a human reviewer."""
    adapter = TextFileAdapter(tmp_path / "artifacts")
    generator = EchoGenerator()

    loop = Loop(adapter=adapter, generator=generator, session_id="readable")
    loop.propose("Nassau market at noon")
    loop.review(Verdict.REJECT, notes="too generic — wanted specific stalls")
    loop.propose("Nassau market at noon with conch sellers and a fortune teller")
    loop.review(Verdict.ACCEPT, notes="warm and specific; ships the brief")

    log_path = tmp_path / "log.yaml"
    write_log(loop.log, log_path)
    content = log_path.read_text(encoding="utf-8")

    # session-level metadata is visible
    assert "session_id: readable" in content
    assert "artifact_type: text" in content
    assert "determinism_tier: semantic_b" in content

    # human-readable verdicts and notes
    assert "verdict: reject" in content
    assert "verdict: accept" in content
    assert "too generic" in content
    assert "warm and specific" in content
