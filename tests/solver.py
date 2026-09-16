"""Whether this machine can plate solve, and why not when it cannot.

Not a test module. `test_solving.py` and `test_processing.py` both need the
same guard, and a guard that drifts between them would let one file silently
stop running.
"""

import shutil
from pathlib import Path

import pytest

#: `funpack` matters as much as the solver: `get_solve_field` unpacks a
#: compressed input before handing it over, and without it every solve of a
#: `.fz` fails with "no WCS header present" while uncompressed frames still
#: work -- which reads as a fixture problem rather than a missing tool.
REQUIRED_TOOLS = ("solve-field", "funpack")

#: Where astrometry.net keeps its configuration, across the packagings we care
#: about: Debian, Homebrew, a source build.
CONFIG_PATHS = (
    "/etc/astrometry.cfg",
    "/usr/local/etc/astrometry.cfg",
    "/opt/homebrew/etc/astrometry.cfg",
    "/usr/share/astrometry/astrometry.cfg",
)


def has_index_files() -> bool:
    """True if any index file is installed where the solver will look.

    The binary being present says nothing about whether it can solve: index
    files are a separate package, and without them `solve-field` runs happily
    and finds nothing. Any index counts rather than a specific one, because
    which index solves depends on field scale -- a full frame here solves with
    `index-4112` and the rebinned fixture with `index-4116`.
    """
    for config in CONFIG_PATHS:
        path = Path(config)
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip().startswith("add_path"):
                directory = Path(line.split(maxsplit=1)[1].strip())
                if any(directory.glob("index-*.fits")):
                    return True
    return False


def _missing() -> list[str]:
    tools = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    return tools if tools else ([] if has_index_files() else ["index files"])


MISSING = _missing()

needs_solver = pytest.mark.skipif(bool(MISSING), reason=f"not available: {', '.join(MISSING)}")
