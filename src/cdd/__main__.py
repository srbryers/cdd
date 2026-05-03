"""CDD CLI entrypoint.

The full CLI (`cdd init`, `cdd review`, `cdd replay`, `cdd diff`) lands
in v0.1.0. This stub exists so `python3 -m cdd` resolves and so the
console_script entry in pyproject.toml has a target.
"""

from __future__ import annotations

import sys

from cdd import __version__


def main() -> int:
    print(f"cdd v{__version__} — Critique-Driven Development")
    print("CLI lands in v0.3.0. See https://github.com/srbryers/cdd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
