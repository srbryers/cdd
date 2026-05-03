"""Protocols for CDD adapters and generators.

CDD separates two concerns:

  - **Generator** owns the connection to an AI provider. It receives
    a prompt and produces raw output. It knows about API endpoints,
    rate limits, and model versions — but knows nothing about what
    kind of artifact is being produced.

  - **Adapter** owns the artifact type. It knows how to persist raw
    generator output as a reference, how to reload it, and what
    determinism tier the artifact targets — but knows nothing about
    which AI provider produced it.

The Loop (cdd.loop) wires the two together. This separation means a
single Generator (e.g., Anthropic) can serve many Adapters (text,
prose, dialogue, structured JSON) and a single Adapter (e.g., image)
can be backed by many Generators (fal.ai, OpenAI DALL-E, etc.).

These are runtime-checkable protocols (not abstract base classes) so
adapters and generators don't need to inherit from CDD types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from cdd.types import DeterminismTier, ModelIdentity


@runtime_checkable
class Adapter(Protocol):
    """An adapter for one artifact type.

    The adapter is the bridge between an AI provider's raw output and
    a CDD log entry's `output_ref`. It declares its artifact type,
    its determinism tier, and how to persist + load.

    Implementations do NOT need to inherit from this Protocol — duck
    typing suffices, and `isinstance(obj, Adapter)` works at runtime
    because of `@runtime_checkable`.

    Type contract: `persist` accepts whatever raw form the paired
    Generator produces (string for text, bytes for images, etc.).
    Adapters should raise `TypeError` if they receive a form they
    can't handle. This is enforced at runtime, not compile time.
    """

    @property
    def artifact_type(self) -> str:
        """Stable identifier for this adapter.

        Recorded in log metadata. Examples: ``"text"``, ``"image"``,
        ``"vegetation_density"``, ``"dialogue_line"``. Use snake_case;
        keep stable across versions.
        """
        ...

    @property
    def determinism_tier(self) -> DeterminismTier:
        """The replay-fidelity guarantee this adapter targets.

        See :class:`cdd.types.DeterminismTier` for the three tiers.
        Adapters should pick the tightest tier their backing generator
        and persistence scheme can honestly support.
        """
        ...

    def persist(self, raw_output: Any, *, namespace: str) -> str:
        """Persist generator output and return an `output_ref`.

        The `namespace` is typically the loop session ID; adapters use
        it to organize artifacts (one directory per loop session, one
        prefix per session, etc.).

        The returned string is a stable reference — adapters should
        guarantee that ``self.load(self.persist(x, namespace=ns)) == x``
        within their determinism tier.

        Raises:
            TypeError: if `raw_output` is the wrong type for this adapter.
        """
        ...

    def load(self, output_ref: str) -> Any:
        """Load an artifact from a previously-persisted `output_ref`.

        Used by replay tooling and by code that consumes accepted
        outputs.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """An interface to an AI provider that produces raw output.

    Generators produce raw output for a prompt. Adapters then persist
    the output and CDD logs the iteration. This is intentionally the
    only thing a generator does — no review, no logging, no artifact
    knowledge.

    Implementations do NOT need to inherit from this Protocol.
    """

    @property
    def model_identity(self) -> ModelIdentity:
        """Identifies the model this generator currently wraps.

        Recorded in every :class:`cdd.types.LogEntry`. May change
        between calls if the generator transparently switches models
        (e.g., for fallback) — each call's identity should be the one
        actually used for that call, captured at call time.
        """
        ...

    @property
    def supports_seed(self) -> bool:
        """Whether this generator honors the ``seed`` parameter.

        Most LLM providers do not honor seeds even when they accept the
        parameter. If False, :class:`cdd.types.DeterminismTier.STRICT_A`
        replay is impossible with this generator and the Loop should
        warn or refuse strict mode.
        """
        ...

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Produce raw output for a prompt.

        Returns whatever raw form this generator produces: a string for
        text generators, bytes for image generators, a dict for
        structured generators. The paired Adapter persists it.

        Args:
            prompt: The actual prompt text sent to the provider.
            seed: Optional seed; only meaningful if `supports_seed` is True.
            extra_params: Provider-specific parameters (temperature,
                top_p, image dimensions, etc.).

        Raises:
            Exception: any provider-side failure surfaces unchanged so
                the Loop can record it in the log.
        """
        ...
