"""CDD log — the chain of LogEntry instances that records authorship.

A Log is one critique session. It owns:
  - session-level metadata (artifact type, determinism tier, session id)
  - the ordered chain of :class:`cdd.types.LogEntry` instances
  - YAML serialization for human-readable persistence

The Log validates structural invariants on append:
  - iteration numbers are strictly 1, 2, 3, ...
  - parent_id of entry N points to entry N-1
  - entry.id is unique within the chain and not equal to its own parent_id
  - cannot append past a resolved (ACCEPT or ABANDON) session

When a log is read from disk via :func:`read_log`, every entry is
re-appended through :meth:`Log.append` so chain validation runs on
load — a hand-edited or corrupted YAML cannot produce a Log that
silently disagrees with its own invariants.

Logs are append-only by convention (mutating past entries breaks the
authorship trail). The dataclass is not frozen because we append
during a session, but read code should never mutate ``entries``.

YAML format is human-editable on purpose — readers will inspect old
logs to understand authorship decisions.

Schema version uses SemVer-style ``MAJOR.MINOR``:
  - same MAJOR is required (forward and backward read works)
  - higher MINOR is read with a warning (additive fields tolerated)
  - lower MINOR reads cleanly (this CDD speaks the older shape too)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict

_LOG_SCHEMA_VERSION = "1.0"
"""Bumped per SemVer rules:

  - MAJOR bump: breaking change to the schema (requires a migrator).
  - MINOR bump: additive change (older readers ignore unknown fields,
    newer readers warn when reading older logs).

Internal — not part of the public API. Migration tooling and tests
import it directly; consumers should not."""


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

    cdd_log_version: str = _LOG_SCHEMA_VERSION
    """SemVer ``MAJOR.MINOR`` schema version. Set automatically on new
    logs; preserved on round-trip read."""

    parent_session_id: str | None = None
    """Optional pointer to a previous session whose accepted artifact
    this session revises. Enables multi-session authorship lineage
    without breaking the per-session append-only chain. None for
    sessions that aren't revisions."""

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

        Invariants enforced:
          - cannot append past a resolved session
          - iteration must equal ``len(entries) + 1``
          - parent_id must equal the previous entry's id (or None for first)
          - entry.id must not already exist in the chain (uniqueness;
            this also forbids self-parent loops, since any reachable
            self-parent is necessarily a duplicate)

        Raises:
            LogConsistencyError: if any invariant is violated.
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

        if any(e.id == entry.id for e in self.entries):
            raise LogConsistencyError(
                f"duplicate entry id within chain: {entry.id!r}"
            )

        self.entries.append(entry)


# ---------- schema versioning helpers ----------


def _parse_schema_version(raw: Any) -> tuple[int, int]:
    """Parse ``MAJOR.MINOR`` into a tuple of ints.

    Raises :class:`LogSchemaError` for any malformed value.
    """
    if not isinstance(raw, str):
        raise LogSchemaError(
            f"cdd_log_version must be a string in MAJOR.MINOR form, "
            f"got {type(raw).__name__}: {raw!r}"
        )
    parts = raw.split(".")
    if len(parts) != 2:
        raise LogSchemaError(
            f"cdd_log_version must be in MAJOR.MINOR form, got {raw!r}"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise LogSchemaError(
            f"cdd_log_version must be MAJOR.MINOR with integer parts, got {raw!r}"
        ) from exc


# ---------- serialization ----------


def to_dict(log: Log) -> dict[str, Any]:
    """Serialize a Log to a plain-dict form suitable for YAML / JSON.

    Round-trips losslessly via :func:`from_dict` when the schema is
    unchanged, and forward-tolerantly across MINOR version bumps.

    This function is importable but not part of the top-level public
    API; consumers typically use :func:`read_log` / :func:`write_log`.
    """
    out: dict[str, Any] = {
        "cdd_log_version": log.cdd_log_version,
        "session_id": log.session_id,
        "artifact_type": log.artifact_type,
        "determinism_tier": log.determinism_tier.value,
        "created_at": log.created_at.isoformat(),
    }
    if log.parent_session_id is not None:
        out["parent_session_id"] = log.parent_session_id
    out["entries"] = [_entry_to_dict(e) for e in log.entries]
    return out


def from_dict(data: Any) -> Log:
    """Deserialize a Log from a plain-dict form (the inverse of :func:`to_dict`).

    Replays each entry through :meth:`Log.append`, so any chain
    inconsistency in the source raises :class:`LogConsistencyError`
    rather than producing a silently-broken Log.

    Raises:
        LogSchemaError: if cdd_log_version is missing, malformed, or
            of an unsupported MAJOR version.
        LogConsistencyError: if the loaded entries fail chain
            validation (corrupt parent_id, iteration skip, etc.).
        KeyError: on missing required top-level fields.
    """
    if not isinstance(data, dict):
        raise LogSchemaError(
            f"expected a YAML mapping at top level, got {type(data).__name__}; "
            "this does not appear to be a CDD log file"
        )

    raw_version = data.get("cdd_log_version")
    if raw_version is None:
        raise LogSchemaError(
            "missing cdd_log_version — this does not appear to be a CDD log file"
        )

    file_major, file_minor = _parse_schema_version(raw_version)
    expected_major, expected_minor = _parse_schema_version(_LOG_SCHEMA_VERSION)

    if file_major != expected_major:
        raise LogSchemaError(
            f"unsupported cdd_log_version major: {raw_version!r} "
            f"(this CDD speaks {_LOG_SCHEMA_VERSION}; major bump indicates "
            "a migration is required)"
        )

    if file_minor > expected_minor:
        warnings.warn(
            f"log written with cdd_log_version {raw_version} but this CDD "
            f"speaks {_LOG_SCHEMA_VERSION}; new fields may be silently dropped",
            stacklevel=3,
        )

    log = Log(
        cdd_log_version=raw_version,
        session_id=data["session_id"],
        artifact_type=data["artifact_type"],
        determinism_tier=DeterminismTier(data["determinism_tier"]),
        created_at=datetime.fromisoformat(data["created_at"]),
        parent_session_id=data.get("parent_session_id"),
    )
    for entry_data in data.get("entries") or []:
        log.append(_entry_from_dict(entry_data))
    return log


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
        LogSchemaError: if cdd_log_version is missing, malformed, or
            of an unsupported MAJOR version (or the file isn't a
            mapping at the top level — empty file, scalar, list).
        LogConsistencyError: if the persisted chain is internally
            inconsistent (corrupt parent_id, iteration skip, etc.).
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
