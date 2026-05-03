"""Tests for cdd.log — Log chain, validation, and YAML round-trip."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cdd.log import (
    Log,
    LogConsistencyError,
    LogSchemaError,
    _LOG_SCHEMA_VERSION,
    from_dict,
    read_log,
    to_dict,
    write_log,
)
from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict


def _entry(
    *,
    id: str,
    parent_id: str | None,
    iteration: int,
    verdict: Verdict = Verdict.REJECT,
    notes: str = "",
) -> LogEntry:
    return LogEntry(
        id=id,
        parent_id=parent_id,
        iteration=iteration,
        timestamp=datetime(2026, 5, 3, 12, 0, iteration, tzinfo=UTC),
        prompt=f"iteration {iteration} prompt",
        seed=42 + iteration,
        model=ModelIdentity(provider="anthropic", model="claude-opus-4-7"),
        extra_params={"temperature": 0.7},
        output_ref=f"artifacts/{id}.txt",
        verdict=verdict,
        notes=notes,
    )


def _empty_log() -> Log:
    return Log(
        session_id="loop-42",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC),
    )


# ---------- chain navigation ----------


def test_empty_log_latest_is_none() -> None:
    log = _empty_log()
    assert log.latest() is None
    assert log.accepted_entry() is None
    assert log.is_resolved() is False


def test_latest_returns_last_appended() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    log.append(_entry(id="b", parent_id="a", iteration=2))
    assert log.latest().id == "b"  # type: ignore[union-attr]


def test_accepted_entry_returns_accept_entry() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.REJECT))
    log.append(_entry(id="b", parent_id="a", iteration=2, verdict=Verdict.ACCEPT))
    accepted = log.accepted_entry()
    assert accepted is not None
    assert accepted.id == "b"


def test_is_resolved_after_accept() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.ACCEPT))
    assert log.is_resolved() is True


def test_is_resolved_after_abandon() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.ABANDON))
    assert log.is_resolved() is True


# ---------- append validation ----------


def test_append_first_entry_requires_no_parent() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    assert len(log.entries) == 1


def test_append_first_entry_with_parent_id_raises() -> None:
    log = _empty_log()
    with pytest.raises(LogConsistencyError, match="parent_id"):
        log.append(_entry(id="a", parent_id="something", iteration=1))


def test_append_second_entry_must_reference_first_as_parent() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    with pytest.raises(LogConsistencyError, match="parent_id"):
        log.append(_entry(id="b", parent_id=None, iteration=2))


def test_append_iteration_must_be_strictly_next() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    with pytest.raises(LogConsistencyError, match="iteration"):
        log.append(_entry(id="b", parent_id="a", iteration=3))


def test_append_after_accept_raises() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.ACCEPT))
    with pytest.raises(LogConsistencyError, match="resolved"):
        log.append(_entry(id="b", parent_id="a", iteration=2))


def test_append_after_abandon_raises() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.ABANDON))
    with pytest.raises(LogConsistencyError, match="resolved"):
        log.append(_entry(id="b", parent_id="a", iteration=2))


def test_append_rejects_duplicate_id() -> None:
    """Re-using an entry id within the chain breaks parent-pointer integrity.

    Constructed with three entries so the duplicate is NOT the immediate
    parent — that way it tests the duplicate-id check in isolation, not
    the parent_id-mismatch check.
    """
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    log.append(_entry(id="b", parent_id="a", iteration=2))
    log.append(_entry(id="c", parent_id="b", iteration=3))
    with pytest.raises(LogConsistencyError, match="duplicate"):
        log.append(_entry(id="a", parent_id="c", iteration=4))


# ---------- to_dict / from_dict round-trip ----------


def test_dict_round_trip_preserves_all_fields() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.REJECT))
    log.append(
        _entry(id="b", parent_id="a", iteration=2, verdict=Verdict.ACCEPT, notes="lyrical")
    )

    data = to_dict(log)
    restored = from_dict(data)

    assert restored.session_id == log.session_id
    assert restored.artifact_type == log.artifact_type
    assert restored.determinism_tier == log.determinism_tier
    assert restored.created_at == log.created_at
    assert restored.cdd_log_version == _LOG_SCHEMA_VERSION
    assert len(restored.entries) == 2
    assert restored.entries[0] == log.entries[0]
    assert restored.entries[1] == log.entries[1]


def test_to_dict_emits_schema_version_as_semver_string() -> None:
    log = _empty_log()
    data = to_dict(log)
    assert data["cdd_log_version"] == _LOG_SCHEMA_VERSION
    assert isinstance(data["cdd_log_version"], str)
    assert "." in data["cdd_log_version"]


def test_from_dict_rejects_major_version_mismatch() -> None:
    log = _empty_log()
    data = to_dict(log)
    data["cdd_log_version"] = "999.0"
    with pytest.raises(LogSchemaError, match="999"):
        from_dict(data)


def test_from_dict_rejects_non_string_version() -> None:
    """Old int-shaped versions (pre-SemVer) must raise a clear error."""
    log = _empty_log()
    data = to_dict(log)
    data["cdd_log_version"] = 1  # int, not "1.0"
    with pytest.raises(LogSchemaError, match="MAJOR.MINOR"):
        from_dict(data)


def test_from_dict_rejects_malformed_version() -> None:
    log = _empty_log()
    data = to_dict(log)
    data["cdd_log_version"] = "totally-not-semver"
    with pytest.raises(LogSchemaError, match="MAJOR.MINOR"):
        from_dict(data)


def test_from_dict_higher_minor_version_warns_but_succeeds() -> None:
    """Forward-compat: read a log written with a newer MINOR version."""
    log = _empty_log()
    data = to_dict(log)
    data["cdd_log_version"] = "1.99"
    with pytest.warns(UserWarning, match="silently dropped"):
        restored = from_dict(data)
    assert restored.cdd_log_version == "1.99"


def test_from_dict_missing_version_key_raises() -> None:
    log = _empty_log()
    data = to_dict(log)
    del data["cdd_log_version"]
    with pytest.raises(LogSchemaError, match="missing"):
        from_dict(data)


def test_from_dict_non_mapping_top_level_raises() -> None:
    """A YAML scalar or list at the top level isn't a CDD log."""
    with pytest.raises(LogSchemaError, match="mapping"):
        from_dict("just a string")  # type: ignore[arg-type]
    with pytest.raises(LogSchemaError, match="mapping"):
        from_dict([])  # type: ignore[arg-type]
    with pytest.raises(LogSchemaError, match="mapping"):
        from_dict(None)  # type: ignore[arg-type]


def test_from_dict_replays_chain_validation_corrupt_parent() -> None:
    """A hand-edited YAML with a corrupt parent_id chain must raise on read."""
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    log.append(_entry(id="b", parent_id="a", iteration=2))
    data = to_dict(log)
    data["entries"][1]["parent_id"] = "wrong-parent"
    with pytest.raises(LogConsistencyError, match="parent_id"):
        from_dict(data)


def test_from_dict_replays_chain_validation_iteration_skip() -> None:
    """Iteration [1, 3] (missing 2) must raise on read, not silently load."""
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    log.append(_entry(id="b", parent_id="a", iteration=2))
    data = to_dict(log)
    data["entries"][1]["iteration"] = 3
    with pytest.raises(LogConsistencyError, match="iteration"):
        from_dict(data)


def test_from_dict_handles_missing_optional_notes() -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1))
    data = to_dict(log)
    del data["entries"][0]["notes"]
    restored = from_dict(data)
    assert restored.entries[0].notes == ""


# ---------- parent_session_id (lineage) ----------


def test_parent_session_id_default_is_none() -> None:
    log = _empty_log()
    assert log.parent_session_id is None


def test_parent_session_id_round_trips() -> None:
    log = Log(
        session_id="rev-2",
        artifact_type="text",
        determinism_tier=DeterminismTier.SEMANTIC_B,
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
        parent_session_id="rev-1",
    )
    data = to_dict(log)
    assert data["parent_session_id"] == "rev-1"
    restored = from_dict(data)
    assert restored.parent_session_id == "rev-1"


def test_parent_session_id_omitted_when_none() -> None:
    """Don't pollute YAML output with `parent_session_id: null` for the common case."""
    log = _empty_log()
    data = to_dict(log)
    assert "parent_session_id" not in data


# ---------- file round-trip ----------


def test_write_and_read_log(tmp_path: Path) -> None:
    log = _empty_log()
    log.append(_entry(id="a", parent_id=None, iteration=1, verdict=Verdict.REJECT))
    log.append(_entry(id="b", parent_id="a", iteration=2, verdict=Verdict.ACCEPT))

    log_path = tmp_path / "session.cdd.yaml"
    write_log(log, log_path)
    assert log_path.exists()

    restored = read_log(log_path)
    assert restored.session_id == log.session_id
    assert len(restored.entries) == 2
    assert restored.entries[1].verdict is Verdict.ACCEPT


def test_write_log_creates_parent_directories(tmp_path: Path) -> None:
    log = _empty_log()
    log_path = tmp_path / "deeply" / "nested" / "dir" / "session.cdd.yaml"
    write_log(log, log_path)
    assert log_path.exists()


def test_yaml_output_is_human_readable(tmp_path: Path) -> None:
    """The whole point of YAML over JSON — humans should be able to read it."""
    log = _empty_log()
    log.append(
        _entry(
            id="a",
            parent_id=None,
            iteration=1,
            notes="this prompt drifted from the brief",
        )
    )
    log_path = tmp_path / "out.yaml"
    write_log(log, log_path)

    content = log_path.read_text(encoding="utf-8")
    assert "session_id: loop-42" in content
    assert "artifact_type: text" in content
    assert "this prompt drifted from the brief" in content


def test_read_log_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_log(tmp_path / "does-not-exist.yaml")


def test_read_log_empty_file_raises_schema_error(tmp_path: Path) -> None:
    """An empty YAML file is not a CDD log — fail with the right exception type."""
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(LogSchemaError):
        read_log(p)


def test_read_log_null_yaml_raises_schema_error(tmp_path: Path) -> None:
    """A file containing only `null` is not a CDD log."""
    p = tmp_path / "null.yaml"
    p.write_text("null\n", encoding="utf-8")
    with pytest.raises(LogSchemaError):
        read_log(p)


def test_read_log_scalar_yaml_raises_schema_error(tmp_path: Path) -> None:
    """A YAML scalar at the top level is not a CDD log."""
    p = tmp_path / "scalar.yaml"
    p.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(LogSchemaError):
        read_log(p)
