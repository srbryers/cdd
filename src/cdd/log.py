"""CDD log — the chain of LogEntry instances that records authorship.

A Log is one critique session. It owns:
  - session-level metadata (artifact type, determinism tier, session id)
  - the ordered chain of :class:`cdd.types.LogEntry` instances
  - YAML serialization for human-readable persistence

The Log validates structural invariants on append:
  - iteration numbers are strictly 1, 2, 3, ...
  - parent_id of entry N points to entry N-1
  - artifact_type matches the session declaration

Logs are append-only by convention (mutating past entries breaks the
authorship trail). The dataclass is not frozen because we append
during a session, but read code should never mutate ``entries``.

YAML format is human-editable on purpose — readers will inspect old
logs to understand authorship decisions. Multi-line prompts read
cleanly; comments are preserved on round-trip via the canonical
serializer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict

LOG_SCHEMA_VERSION = 1
"""Bumped when the YAML schema breaks backward compatibility.

Increment requires a migrator. Logs of older versions can still be
read but may need transformation."""


class LogSchemaError(ValueError):
    """Raised when reading a log with a schema CDD does not understand."""


class LogConsistencyError(ValueError):
    """Raised when an append would break log invariants."""


@dataclass
class Log:
    """A chain of :class:`cdd.types.LogEntry` instances forming one session.

    A session is one human's iterative dialog with one AI generator
    over one artifact type. The session ends when an entry's verdict
    is :attr:`cdd.types.Verdict.ACCEPT` or :attr:`cdd.types.Verdict.ABANDON`.
    """

    session_id: str
    """Unique identifier for this session (typically UUID4 string)."""

    artifact_type: str
    """The :attr:`cdd.protocols.Adapter.artifact_type` of the adapter
    that handled this session. Pinned at session start; entries do
    not record their own artifact_type."""

    determinism_tier: DeterminismTier
    """The adapter's declared determinism tier at session start."""

    created_at: datetime
    """When the session started (timezone-aware, UTC recommended)."""

    entries: list[LogEntry] = field(default_factory=list)
    """The chain of entries, oldest first. Append-only by convention."""

    cdd_log_version: int = LOG_SCHEMA_VERSION
    """Schema version of this log. Set automatically; preserve on read."""

    # ---------- chain navigation ----------

    def latest(self) -> LogEntry | None:
        """Last entry in the chain, or None for an empty log."""
        return self.entries[-1] if self.entries else None

    def accepted_entry(self) -> LogEntry | None:
        """The :attr:`Verdict.ACCEPT` entry if the session resolved that
        way, else None.

        A session has at most one ACCEPT entry — once accepted, the
        session ends.
        """
        for entry in self.entries:
            if entry.verdict is Verdict.ACCEPT:
                return entry
        return None

    def is_resolved(self) -> bool:
        """True if the session has ended (ACCEPT or ABANDON)."""
        last = self.latest()
        if last is None:
            return False
        return last.verdict in (Verdict.ACCEPT, Verdict.ABANDON)

    # ---------- mutation ----------

    def append(self, entry: LogEntry) -> None:
        """Add an entry to the chain, validating consistency.

        Raises:
            LogConsistencyError: if iteration number is wrong, parent_id
                doesn't match, or appending past a resolved session.
        """
        if self.is_resolved():
            raise LogConsistencyError(
                "cannot append to a resolved session "
                f"(latest verdict: {self.latest().verdict.value})"  # type: ignore[union-attr]
            )

        expected_iteration = len(self.entries) + 1
        if entry.iteration != expected_iteration:
            raise LogConsistencyError(
                f"expected iteration {expected_iteration}, got {entry.iteration}"
            )

        expected_parent_id = self.entries[-1].id if self.entries else None
        if entry.parent_id != expected_parent_id:
            raise LogConsistencyError(
                f"parent_id mismatch: expected {expected_parent_id!r}, "
                f"got {entry.parent_id!r}"
            )

        self.entries.append(entry)


# ---------- serialization ----------


def to_dict(log: Log) -> dict[str, Any]:
    """Serialize a Log to a plain-dict form suitable for YAML / JSON.

    Round-trips losslessly via :func:`from_dict`.
    """
    return {
        "cdd_log_version": log.cdd_log_version,
        "session_id": log.session_id,
        "artifact_type": log.artifact_type,
        "determinism_tier": log.determinism_tier.value,
        "created_at": log.created_at.isoformat(),
        "entries": [_entry_to_dict(e) for e in log.entries],
    }


def from_dict(data: dict[str, Any]) -> Log:
    """Deserialize a Log from a plain-dict form (the inverse of :func:`to_dict`).

    Raises:
        LogSchemaError: if cdd_log_version is unsupported.
        KeyError / ValueError: on missing or malformed fields.
    """
    version = data.get("cdd_log_version")
    if version != LOG_SCHEMA_VERSION:
        raise LogSchemaError(
            f"unsupported cdd_log_version: {version!r} "
            f"(this CDD speaks version {LOG_SCHEMA_VERSION})"
        )

    return Log(
        cdd_log_version=version,
        session_id=data["session_id"],
        artifact_type=data["artifact_type"],
        determinism_tier=DeterminismTier(data["determinism_tier"]),
        created_at=datetime.fromisoformat(data["created_at"]),
        entries=[_entry_from_dict(e) for e in data["entries"]],
    )


def write_log(log: Log, path: Path | str) -> None:
    """Write a Log as YAML to the given path.

    Creates parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            to_dict(log),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def read_log(path: Path | str) -> Log:
    """Read a Log from a YAML file written by :func:`write_log`.

    Raises:
        LogSchemaError: if cdd_log_version is unsupported.
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return from_dict(data)


# ---------- entry serialization helpers ----------


def _entry_to_dict(entry: LogEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "parent_id": entry.parent_id,
        "iteration": entry.iteration,
        "timestamp": entry.timestamp.isoformat(),
        "prompt": entry.prompt,
        "seed": entry.seed,
        "model": {
            "provider": entry.model.provider,
            "model": entry.model.model,
            "version": entry.model.version,
        },
        "extra_params": dict(entry.extra_params),
        "output_ref": entry.output_ref,
        "verdict": entry.verdict.value,
        "notes": entry.notes,
    }


def _entry_from_dict(data: dict[str, Any]) -> LogEntry:
    model_data = data["model"]
    return LogEntry(
        id=data["id"],
        parent_id=data["parent_id"],
        iteration=data["iteration"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        prompt=data["prompt"],
        seed=data["seed"],
        model=ModelIdentity(
            provider=model_data["provider"],
            model=model_data["model"],
            version=model_data.get("version"),
        ),
        extra_params=dict(data.get("extra_params") or {}),
        output_ref=data["output_ref"],
        verdict=Verdict(data["verdict"]),
        notes=data.get("notes", ""),
    )
