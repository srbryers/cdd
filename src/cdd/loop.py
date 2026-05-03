"""CDD Loop — the orchestrator for one critique session.

A Loop wires together an :class:`cdd.protocols.Adapter` (artifact
type) and a :class:`cdd.protocols.Generator` (AI provider) and
records each iteration as a :class:`cdd.types.LogEntry`.

Lifecycle:

  1. :meth:`Loop.propose` — generator produces output, adapter persists
     it, Loop holds the result as *pending* awaiting review.
  2. :meth:`Loop.review` — human's verdict finalizes the pending
     proposal into a logged entry. If the verdict is ACCEPT or
     ABANDON, the session is resolved and no further proposals are
     accepted.

A Loop holds at most one pending proposal at a time. You must
:meth:`review` a pending proposal before issuing the next
:meth:`propose`. This is the discipline — every generated output is
either accepted, rejected, refined, or abandoned. Nothing slips
through unreviewed.

Resuming an in-progress session:

    ``Loop(adapter=..., generator=..., log=read_log(path))``

picks up where a previous session left off, provided the loaded log
is not yet resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from cdd.log import Log
from cdd.protocols import Adapter, Generator
from cdd.types import LogEntry, ModelIdentity, Verdict


class LoopStateError(RuntimeError):
    """Raised when the Loop is asked to do something its state forbids.

    For example: proposing while a previous proposal is pending review,
    reviewing with no pending proposal, or proposing after the session
    has been resolved.
    """


@dataclass
class _Pending:
    """Internal state for a generated-but-not-yet-reviewed proposal."""

    prompt: str
    raw_output: Any
    output_ref: str
    seed: int | None
    model: ModelIdentity
    extra_params: dict[str, Any] = field(default_factory=dict)


class Loop:
    """One CDD critique session.

    The Loop is the only public type that mutates state; everything it
    persists (LogEntry, Log) is otherwise immutable or append-only.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        generator: Generator,
        session_id: str | None = None,
        log: Log | None = None,
    ) -> None:
        """Construct a new Loop, or resume an existing one.

        Args:
            adapter: The artifact-type adapter for this session.
            generator: The AI provider generator for this session.
            session_id: Optional session ID for new sessions; auto-generated
                via UUID4 if omitted. Ignored when ``log`` is provided.
            log: Optional existing log to resume. Must not be resolved.
                Its artifact_type and determinism_tier must match the
                supplied adapter.

        Raises:
            LoopStateError: if resuming a resolved log, or if the log
                disagrees with the adapter on artifact_type or tier.
        """
        if log is not None:
            if log.is_resolved():
                raise LoopStateError(
                    "cannot resume a resolved log "
                    f"(latest verdict: {log.latest().verdict.value})"  # type: ignore[union-attr]
                )
            if log.artifact_type != adapter.artifact_type:
                raise LoopStateError(
                    f"log artifact_type {log.artifact_type!r} does not match "
                    f"adapter {adapter.artifact_type!r}"
                )
            if log.determinism_tier != adapter.determinism_tier:
                raise LoopStateError(
                    f"log determinism_tier {log.determinism_tier.value!r} does not match "
                    f"adapter {adapter.determinism_tier.value!r}"
                )
            if session_id is not None and session_id != log.session_id:
                raise LoopStateError(
                    f"explicit session_id {session_id!r} disagrees with "
                    f"log.session_id {log.session_id!r}; pass one or the other"
                )
            self._log = log
        else:
            self._log = Log(
                session_id=session_id or str(uuid4()),
                artifact_type=adapter.artifact_type,
                determinism_tier=adapter.determinism_tier,
                created_at=datetime.now(UTC),
            )

        self._adapter = adapter
        self._generator = generator
        self._pending: _Pending | None = None

    # ---------- read-only accessors ----------

    @property
    def log(self) -> Log:
        """The :class:`cdd.log.Log` this Loop appends to."""
        return self._log

    @property
    def is_pending(self) -> bool:
        """True if a proposal awaits review."""
        return self._pending is not None

    @property
    def is_resolved(self) -> bool:
        """True if the session has ended (ACCEPT or ABANDON entry logged)."""
        return self._log.is_resolved()

    def current_output(self) -> Any:
        """The pending proposal's raw output, or None if no proposal is pending."""
        return self._pending.raw_output if self._pending is not None else None

    # ---------- session operations ----------

    def propose(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Generate output for a prompt and stage it for review.

        Returns the raw output (so the caller can present it to the
        human). The output has also been persisted via the adapter at
        this point; the persisted reference is captured internally and
        will appear in the LogEntry once :meth:`review` is called.

        Args:
            prompt: The prompt sent to the generator.
            seed: Optional seed; only honored if ``generator.supports_seed``.
            extra_params: Optional generator-specific params (temperature,
                top_p, image dimensions, etc.).

        Returns:
            The raw output produced by the generator.

        Raises:
            LoopStateError: if the session is resolved, or if a previous
                proposal still awaits review.
        """
        if self.is_resolved:
            raise LoopStateError("session is resolved; no further proposals accepted")
        if self._pending is not None:
            raise LoopStateError(
                "a proposal is already pending review; call review() first"
            )

        raw_output = self._generator.generate(
            prompt,
            seed=seed,
            extra_params=extra_params,
        )
        output_ref = self._adapter.persist(
            raw_output,
            namespace=self._log.session_id,
        )

        self._pending = _Pending(
            prompt=prompt,
            raw_output=raw_output,
            output_ref=output_ref,
            seed=seed,
            model=self._generator.model_identity,
            extra_params=dict(extra_params or {}),
        )
        return raw_output

    def review(self, verdict: Verdict, *, notes: str = "") -> LogEntry:
        """Finalize the pending proposal into a logged entry.

        Args:
            verdict: The human's judgment.
            notes: Free-text reasoning. Recommended — the published
                authorship trail benefits from explicit reasoning.

        Returns:
            The newly-appended :class:`cdd.types.LogEntry`.

        Raises:
            LoopStateError: if no proposal is pending review.
        """
        if self._pending is None:
            raise LoopStateError("no pending proposal to review; call propose() first")

        prev = self._log.latest()
        entry = LogEntry(
            id=str(uuid4()),
            parent_id=prev.id if prev is not None else None,
            iteration=len(self._log.entries) + 1,
            timestamp=datetime.now(UTC),
            prompt=self._pending.prompt,
            seed=self._pending.seed,
            model=self._pending.model,
            extra_params=dict(self._pending.extra_params),
            output_ref=self._pending.output_ref,
            verdict=verdict,
            notes=notes,
        )
        self._log.append(entry)
        self._pending = None
        return entry
