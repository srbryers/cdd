"""EchoGenerator — a trivial deterministic generator for tests + smoke.

EchoGenerator produces output that's a pure function of (prompt, seed,
extra_params), with no AI calls. It exists for two purposes:

  1. End-to-end testing without API keys, network, or quotas.
  2. Demonstrating the :class:`cdd.protocols.Generator` protocol with
     the smallest possible real implementation.

Because the output is deterministic, EchoGenerator targets
:class:`cdd.types.DeterminismTier.STRICT_A`-compatible flows even
though no real AI is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cdd.types import ModelIdentity


class EchoGenerator:
    """Deterministic generator that echoes the prompt + seed + params.

    The output format is intentionally machine-parseable so tests can
    assert on it precisely::

        echo[seed=<seed>; params=<sorted-kv>]: <prompt>
    """

    supports_seed = True

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider="cdd", model="echo", version="1")

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        params_repr = _format_params(extra_params)
        return f"echo[seed={seed}; params={params_repr}]: {prompt}"


def _format_params(extra_params: Mapping[str, Any] | None) -> str:
    if not extra_params:
        return "{}"
    items = sorted(extra_params.items())
    inner = ", ".join(f"{k}={v!r}" for k, v in items)
    return "{" + inner + "}"
