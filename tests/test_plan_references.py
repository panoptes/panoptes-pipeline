"""Every cross-reference in `plans/` must resolve to a section that exists.

These documents get renumbered as priorities move -- drift was inserted at 3.2
and pushed nine sections down by one -- and a stale "(3.5)" pointing at the
wrong section is the kind of rot nobody notices until it misleads someone. This
test is cheap insurance for a deliverable that is prose rather than code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PLANS = Path(__file__).resolve().parents[1] / "plans"

#: "### 3.2 Drift and sub-pixel sampling" and "## 6. Action items"
SECTION = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)?)[.\s]", re.MULTILINE)
#: "**6.1 -- Pick the first sequence"
ACTION_ITEM = re.compile(r"^\*\*(\d+\.\d+)\s+--", re.MULTILINE)
#: bare "(3.5)" and prefixed "improvement plan 3.4" / "conformance audit 5.0"
BARE_REF = re.compile(r"\((\d+\.\d+)\)")
NAMED_REF = re.compile(r"(improvement plan|conformance audit)\s+(\d+(?:\.\d+)?)")

DOC_FILES = {"improvement plan": "improvement-plan.md", "conformance audit": "conformance-audit.md"}


def sections(name: str) -> set[str]:
    text = (PLANS / DOC_FILES[name]).read_text()
    found = set(SECTION.findall(text)) | set(ACTION_ITEM.findall(text))
    # A reference to a whole chapter ("improvement plan 6") is valid too.
    return found | {number.split(".")[0] for number in found}


@pytest.fixture(scope="module")
def known() -> dict[str, set[str]]:
    return {name: sections(name) for name in DOC_FILES}


def test_plans_are_present():
    for filename in DOC_FILES.values():
        assert (PLANS / filename).exists(), f"{filename} is missing"


def test_improvement_plan_numbering_has_no_gaps(known):
    """A gap almost always means a renumber went wrong halfway."""
    for chapter in ("1", "2", "3", "4"):
        numbers = sorted(
            int(s.split(".")[1]) for s in known["improvement plan"] if s.startswith(f"{chapter}.")
        )
        assert numbers == list(range(1, len(numbers) + 1)), f"chapter {chapter} is {numbers}"


@pytest.mark.parametrize("doc", sorted(DOC_FILES))
def test_named_cross_references_resolve(doc, known):
    text = (PLANS / DOC_FILES[doc]).read_text()
    broken = [
        f"{target} {number}"
        for target, number in NAMED_REF.findall(text)
        if number not in known[target]
    ]
    assert not broken, f"{DOC_FILES[doc]} references sections that do not exist: {broken}"


def test_bare_references_in_the_improvement_plan_resolve(known):
    """A bare "(3.5)" inside the improvement plan means its own section 3.5."""
    text = (PLANS / DOC_FILES["improvement plan"]).read_text()
    broken = [ref for ref in BARE_REF.findall(text) if ref not in known["improvement plan"]]
    assert not broken, f"improvement-plan.md has dangling references: {sorted(set(broken))}"


def test_source_code_references_resolve(known):
    """Docstrings cite these sections too, and drift out of date the same way."""
    root = PLANS.parent
    broken = []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        for target, number in NAMED_REF.findall(path.read_text()):
            if number not in known[target]:
                broken.append(f"{path.relative_to(root)}: {target} {number}")
    assert not broken, f"stale plan references in code: {broken}"
