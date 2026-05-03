"""Reference and built-in CDD adapters.

Adapters here are part of the CDD distribution. Project-specific
adapters (e.g., FATHOMS vegetation density) live in their own repos
and import :class:`cdd.protocols.Adapter` to declare conformance.
"""

from cdd.adapters.text import TextFileAdapter

__all__ = ["TextFileAdapter"]
