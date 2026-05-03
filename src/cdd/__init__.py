"""CDD — Critique-Driven Development.

A methodology for AI-assisted creative work where the human is editorial,
not generative. The AI proposes; the human disposes. Every iteration is
logged. The artifact is reproducible from the log alone.

This package is pre-alpha. The public API surface is stabilizing through
v0.x; expect breaking changes until v1.0. See the README for the
conceptual framing.
"""

from cdd.adapters import TextFileAdapter
from cdd.generators import EchoGenerator
from cdd.log import (
    LOG_SCHEMA_VERSION,
    Log,
    LogConsistencyError,
    LogSchemaError,
    read_log,
    write_log,
)
from cdd.loop import Loop, LoopStateError
from cdd.protocols import Adapter, Generator
from cdd.types import DeterminismTier, LogEntry, ModelIdentity, Verdict

__version__ = "0.0.1"

__all__ = [
    "Adapter",
    "DeterminismTier",
    "EchoGenerator",
    "Generator",
    "LOG_SCHEMA_VERSION",
    "Log",
    "LogConsistencyError",
    "LogEntry",
    "LogSchemaError",
    "Loop",
    "LoopStateError",
    "ModelIdentity",
    "TextFileAdapter",
    "Verdict",
    "__version__",
    "read_log",
    "write_log",
]
