"""Smoke test — package imports cleanly and exposes its version."""

import cdd


def test_package_importable() -> None:
    assert cdd is not None


def test_version_exposed() -> None:
    assert hasattr(cdd, "__version__")
    assert cdd.__version__ == "0.1.0"
