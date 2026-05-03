"""Tests for cdd.generators.echo — the reference EchoGenerator."""

from __future__ import annotations

from cdd.generators.echo import EchoGenerator
from cdd.protocols import Generator
from cdd.types import ModelIdentity


def test_satisfies_generator_protocol() -> None:
    g = EchoGenerator()
    assert isinstance(g, Generator)


def test_model_identity() -> None:
    g = EchoGenerator()
    assert g.model_identity == ModelIdentity(provider="cdd", model="echo", version="1")


def test_supports_seed_true() -> None:
    g = EchoGenerator()
    assert g.supports_seed is True


def test_generate_returns_str_with_prompt() -> None:
    g = EchoGenerator()
    out = g.generate("Caribbean cove at dawn")
    assert "Caribbean cove at dawn" in out
    assert out.startswith("echo[")


def test_generate_includes_seed() -> None:
    g = EchoGenerator()
    out = g.generate("anything", seed=42)
    assert "seed=42" in out


def test_generate_includes_no_params_marker_when_empty() -> None:
    g = EchoGenerator()
    out = g.generate("anything")
    assert "params={}" in out


def test_generate_includes_extra_params_sorted() -> None:
    g = EchoGenerator()
    out = g.generate("anything", extra_params={"top_p": 0.9, "temperature": 0.7})
    # Sorted alphabetically: temperature, then top_p
    assert "temperature=0.7" in out
    assert "top_p=0.9" in out
    assert out.index("temperature") < out.index("top_p")


def test_generate_is_deterministic() -> None:
    """Same args -> same output. Foundational to its 'reference for tests' role."""
    g = EchoGenerator()
    a = g.generate("x", seed=1, extra_params={"k": "v"})
    b = g.generate("x", seed=1, extra_params={"k": "v"})
    assert a == b
