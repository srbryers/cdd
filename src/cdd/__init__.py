"""CDD — Critique-Driven Development.

A methodology for AI-assisted creative work where the human is editorial,
not generative. The AI proposes; the human disposes. Every iteration is
logged. The artifact is reproducible from the log alone.

This package is pre-alpha. The public API surface is stabilizing through
v0.x; expect breaking changes until v1.0. See the README for the
conceptual framing.
"""

from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict

__version__ = "0.0.1"

__all__ = [
    "DeterminismTier",
    "LogEntry",
    "ModelIdentity",
    "Verdict",
    "__version__",
]
