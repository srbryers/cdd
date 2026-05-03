"""Core types for Critique-Driven Development.

These types are the contract for every CDD log entry, every adapter,
and every generator. Intentionally minimal — just enough to capture
an authorship trail that can be replayed.

Public exports:
    Verdict             — the human's judgment on a generated output
    DeterminismTier     — replay-fidelity guarantee an adapter targets
    ModelIdentity       — provider + model + version triple
    LogEntry            — one iteration in a CDD loop
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """The human's judgment on a generated output."""

    ACCEPT = "accept"
    """Output is good. End the loop with this artifact as the result."""

    REJECT = "reject"
    """Output is wrong. Try again with the same prompt and a new seed."""

    REFINE = "refine"
    """Output is close. Try again with a refined prompt — the next entry
    in the chain carries the new prompt."""

    ABANDON = "abandon"
    """Stop the loop entirely. No artifact produced."""


class DeterminismTier(StrEnum):
    """The replay-fidelity guarantee an adapter targets.

    Adapters declare their tier so replay tooling knows what to expect
    when regenerating an artifact from a log.
    """

    STRICT_A = "strict_a"
    """Replay produces byte-identical output. Requires honored seed and
    a pinned model version. Brittle: silently breaks when AI providers
    deprecate or silently update models."""

    SEMANTIC_B = "semantic_b"
    """Replay produces semantically-equivalent output. Re-rolls allowed;
    human re-verifies the result. **Default tier** — survives model
    changes that STRICT_A cannot."""

    BEST_EFFORT_C = "best_effort_c"
    """Replay attempts to recreate; mismatches are logged but not
    treated as failures. Useful for archival and historical reads."""


@dataclass(frozen=True)
class ModelIdentity:
    """Identifies the specific AI model that produced an output.

    Recorded in every LogEntry so replay tooling can:
      - warn when a model is deprecated or unavailable
      - substitute a successor model with re-verification
      - prove provenance of an artifact
    """

    provider: str
    """The vendor/service: "anthropic", "openai", "fal", "local-llama", etc."""

    model: str
    """The model name: "claude-opus-4-7", "gpt-4o", "flux-pro", etc."""

    version: str | None = None
    """Specific version pin if known. None means "whatever was current"
    at generation time — fine for SEMANTIC_B, insufficient for STRICT_A."""

    def __str__(self) -> str:
        if self.version is None:
            return f"{self.provider}/{self.model}"
        return f"{self.provider}/{self.model}@{self.version}"


@dataclass(frozen=True)
class LogEntry:
    """One iteration in a CDD loop.

    Each entry captures one round of: AI generates → human reviews →
    log records. The chain of LogEntry instances IS the authorship
    trail; replaying the chain regenerates the artifact.

    Frozen by design — once logged, an entry is immutable. Mutating
    `extra_params` after construction is convention-banned but not
    runtime-prevented (frozen dataclass freezes the field binding,
    not the dict contents).
    """

    id: str
    """Unique identifier (typically UUID4 string) for this entry."""

    parent_id: str | None
    """The previous entry in this loop's chain. None for the first
    entry of a session."""

    iteration: int
    """1-based iteration number within the session. The first entry is
    iteration 1."""

    timestamp: datetime
    """When this entry was created. Should be timezone-aware (UTC
    recommended)."""

    prompt: str
    """The actual prompt text sent to the generator."""

    seed: int | None
    """The seed used for generation. None if the generator does not
    honor seeds (most LLM providers don't, in practice)."""

    model: ModelIdentity
    """The model that produced output_ref."""

    extra_params: dict[str, Any]
    """Generator-specific parameters: temperature, top_p, stop sequences,
    image dimensions, anything else the generator received. Opaque to
    CDD itself — the adapter and generator interpret these."""

    output_ref: str
    """Reference to the generated output. The adapter interprets this:
    a file path, a content-addressed hash, a URI, etc."""

    verdict: Verdict
    """The human's judgment on this output."""

    notes: str = ""
    """Free-text reasoning the human provided — typically why a verdict
    was issued. Important for replay tooling and the eventual published
    authorship trail."""
