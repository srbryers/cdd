"""Plain-text artifact adapter — the reference adapter.

The simplest possible CDD adapter: persist a string to a UTF-8 file,
load it back. Used as the proof-of-protocol implementation, the
test target for end-to-end Loop integration, and the smoke test for
external CDD consumers.

Persistence is content-addressed: the output_ref is built from a
SHA-256 prefix of the raw output, which means identical generator
outputs deduplicate to the same file. That's a friendly property
for SEMANTIC_B replay where re-rolls may converge.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from cdd.types import DeterminismTier


class TextFileAdapter:
    """Persists text artifacts to a directory tree of UTF-8 files.

    The ``output_ref`` for each persisted output is a relative path of
    the form ``"<namespace>/<sha256-prefix>.txt"`` — a Loop's
    session_id is the typical namespace, so all artifacts for one
    session land under one subdirectory.

    Determinism tier is ``SEMANTIC_B`` by default — adapters can
    subclass and override to declare a different tier when paired
    with a strict-replay-capable generator.
    """

    artifact_type = "text"
    determinism_tier: DeterminismTier = DeterminismTier.SEMANTIC_B

    HASH_PREFIX_LEN = 16
    """How many hex chars of SHA-256 to use in the filename. 16 is
    enough to make collisions extraordinarily unlikely within a single
    session and keeps paths readable."""

    def __init__(self, base_dir: Path | str) -> None:
        """Configure the on-disk root for persisted artifacts.

        Args:
            base_dir: directory under which artifacts are stored.
                Created on first :meth:`persist` if missing.
        """
        self._base = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        """The configured root directory."""
        return self._base

    def persist(self, raw_output: Any, *, namespace: str) -> str:
        """Write ``raw_output`` to ``<base_dir>/<namespace>/<hash>.txt``.

        Returns the relative path (the part after ``base_dir``) as
        the ``output_ref``.

        Raises:
            TypeError: if ``raw_output`` is not a ``str``.
        """
        if not isinstance(raw_output, str):
            raise TypeError(
                f"TextFileAdapter requires a str, got {type(raw_output).__name__}"
            )

        digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        rel_path = f"{namespace}/{digest[: self.HASH_PREFIX_LEN]}.txt"
        full_path = self._base / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(raw_output, encoding="utf-8")
        return rel_path

    def load(self, output_ref: str) -> Any:
        """Read the artifact previously persisted under ``output_ref``."""
        return (self._base / output_ref).read_text(encoding="utf-8")
