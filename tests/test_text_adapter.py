"""Tests for cdd.adapters.text — the reference TextFileAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from cdd.adapters.text import TextFileAdapter
from cdd.protocols import Adapter
from cdd.types import DeterminismTier


def test_satisfies_adapter_protocol(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    assert isinstance(a, Adapter)


def test_artifact_type_is_text(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    assert a.artifact_type == "text"


def test_default_determinism_tier(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    assert a.determinism_tier is DeterminismTier.SEMANTIC_B


def test_persist_creates_file_under_namespace(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    ref = a.persist("a Caribbean cove at dawn", namespace="loop-1")
    assert ref.startswith("loop-1/")
    assert ref.endswith(".txt")
    assert (tmp_path / ref).exists()


def test_persist_load_roundtrip(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    raw = "warm light across the lagoon"
    ref = a.persist(raw, namespace="loop-2")
    assert a.load(ref) == raw


def test_persist_is_content_addressed(tmp_path: Path) -> None:
    """Same input -> same output_ref -> the file is deduped."""
    a = TextFileAdapter(tmp_path)
    ref1 = a.persist("identical content", namespace="loop-3")
    ref2 = a.persist("identical content", namespace="loop-3")
    assert ref1 == ref2


def test_persist_different_inputs_get_different_refs(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    ref1 = a.persist("alpha", namespace="loop-4")
    ref2 = a.persist("beta", namespace="loop-4")
    assert ref1 != ref2


def test_persist_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested"
    a = TextFileAdapter(nested)
    ref = a.persist("any", namespace="x")
    assert (nested / ref).exists()


def test_persist_rejects_non_string(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    with pytest.raises(TypeError, match="str"):
        a.persist(b"raw bytes", namespace="x")
    with pytest.raises(TypeError, match="str"):
        a.persist(42, namespace="x")


def test_unicode_persists_correctly(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    raw = "Vodou ↔ Vaudou ↔ Vodun  (𓂀  ✶  𓂀)"
    ref = a.persist(raw, namespace="x")
    assert a.load(ref) == raw


def test_base_dir_property_is_path(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    assert a.base_dir == Path(tmp_path)


def test_persist_empty_string_round_trips(tmp_path: Path) -> None:
    """Empty string is a valid output (e.g., a deliberately-blank dialogue line)."""
    a = TextFileAdapter(tmp_path)
    ref = a.persist("", namespace="loop-empty")
    assert a.load(ref) == ""


# ---------- path-traversal defenses ----------


def test_persist_rejects_namespace_with_path_separator(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    with pytest.raises(ValueError, match="separator"):
        a.persist("anything", namespace="../sibling")
    with pytest.raises(ValueError, match="separator"):
        a.persist("anything", namespace="nested/sub")
    with pytest.raises(ValueError, match="separator"):
        a.persist("anything", namespace="back\\slash")


def test_persist_rejects_namespace_starting_with_dot(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    with pytest.raises(ValueError, match=r"start with '\.'"):
        a.persist("anything", namespace=".hidden")


def test_persist_rejects_empty_namespace(tmp_path: Path) -> None:
    a = TextFileAdapter(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        a.persist("anything", namespace="")


def test_load_rejects_traversal_output_ref(tmp_path: Path) -> None:
    """A hand-edited or untrusted log cannot escape base_dir on load."""
    a = TextFileAdapter(tmp_path)
    # Write a sibling file outside base_dir to make the test concrete.
    sibling = tmp_path.parent / "sibling.txt"
    sibling.write_text("secret", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="escapes base_dir"):
            a.load("../sibling.txt")
    finally:
        sibling.unlink(missing_ok=True)
