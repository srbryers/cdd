"""Tests for cdd.protocols — Adapter and Generator runtime protocols.

These tests verify that:
  - duck-typed implementations satisfy the protocols at runtime
  - missing required members fail isinstance checks
  - the protocols are stable contracts that real adapters can target
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from cdd.protocols import Adapter, Generator
from cdd.types import DeterminismTier, ModelIdentity


# ---------- Adapter ----------


class _MinimalTextAdapter:
    """Smallest possible adapter implementation for protocol-conformance tests."""

    artifact_type = "text"
    determinism_tier = DeterminismTier.SEMANTIC_B

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def persist(self, raw_output: Any, *, namespace: str) -> str:
        if not isinstance(raw_output, str):
            raise TypeError(f"text adapter requires str, got {type(raw_output).__name__}")
        ref = f"{namespace}/{len(self._store)}"
        self._store[ref] = raw_output
        return ref

    def load(self, output_ref: str) -> Any:
        return self._store[output_ref]


class _IncompleteAdapter:
    """Missing the persist method — must NOT satisfy the protocol."""

    artifact_type = "text"
    determinism_tier = DeterminismTier.SEMANTIC_B

    def load(self, output_ref: str) -> Any:
        return None


def test_minimal_adapter_satisfies_protocol() -> None:
    a = _MinimalTextAdapter()
    assert isinstance(a, Adapter)


def test_incomplete_adapter_fails_isinstance() -> None:
    a = _IncompleteAdapter()
    assert not isinstance(a, Adapter)


def test_adapter_persist_load_roundtrip() -> None:
    a = _MinimalTextAdapter()
    ref = a.persist("a Caribbean dawn", namespace="loop-42")
    assert a.load(ref) == "a Caribbean dawn"


def test_adapter_rejects_wrong_type() -> None:
    a = _MinimalTextAdapter()
    with pytest.raises(TypeError, match="str"):
        a.persist(b"bytes-not-str", namespace="loop-42")


def test_adapter_artifact_type_is_string_identifier() -> None:
    a = _MinimalTextAdapter()
    assert a.artifact_type == "text"
    assert isinstance(a.artifact_type, str)


def test_adapter_determinism_tier_is_enum() -> None:
    a = _MinimalTextAdapter()
    assert a.determinism_tier is DeterminismTier.SEMANTIC_B


# ---------- Generator ----------


class _StubGenerator:
    """Smallest generator: echoes prompt back, no real AI."""

    supports_seed = False

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider="stub", model="echo")

    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        return f"echo: {prompt}"


class _IncompleteGenerator:
    """No generate method — must NOT satisfy the protocol."""

    supports_seed = False

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider="stub", model="echo")


def test_minimal_generator_satisfies_protocol() -> None:
    g = _StubGenerator()
    assert isinstance(g, Generator)


def test_incomplete_generator_fails_isinstance() -> None:
    g = _IncompleteGenerator()
    assert not isinstance(g, Generator)


def test_generator_returns_output_for_prompt() -> None:
    g = _StubGenerator()
    out = g.generate("Caribbean cove at dawn")
    assert out == "echo: Caribbean cove at dawn"


def test_generator_model_identity_is_stable_value() -> None:
    g = _StubGenerator()
    assert g.model_identity == ModelIdentity(provider="stub", model="echo")


def test_generator_supports_seed_is_bool() -> None:
    g = _StubGenerator()
    assert g.supports_seed is False


# ---------- Adapter + Generator integration shape ----------


def test_generator_output_can_be_persisted_by_adapter() -> None:
    """Smoke check: stub generator output flows into minimal adapter cleanly."""
    g = _StubGenerator()
    a = _MinimalTextAdapter()

    raw = g.generate("describe a market in Nassau")
    ref = a.persist(raw, namespace="loop-1")
    loaded = a.load(ref)

    assert loaded == raw
    assert loaded == "echo: describe a market in Nassau"
