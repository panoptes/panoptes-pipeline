# Project standards

How the four repositories of the PANOPTES rebuild are built, linted, tested,
documented and released. Cite as "project standards 3.2".

This document is the reasoning. Whether a repository conforms yet is in the
[issue tracker](https://github.com/panoptes/panoptes-pipeline/issues): the
parent is panoptes/panoptes-pipeline#218, with one issue per repository for the
work outside this one.

The four are [POCS](https://github.com/panoptes/POCS), which writes the FITS
headers; [`panoptes-utils`](https://github.com/panoptes/panoptes-utils), the
shared base; this repository, which processes and produces; and
[`panoptes-data`](https://github.com/panoptes/panoptes-data), which discovers,
fetches and queries. Data contract 8 says who owns what. This document is about
the packaging they share, not the data they exchange.

"Fleet" is not used here for the repositories. In this project it means the
heterogeneous set of units and DSLR bodies, which is load-bearing vocabulary;
reusing it for four git repositories would blunt it.

---

## 1. Why one standard

Not tidiness. Three concrete costs of having four.

**Knowledge does not transfer.** Every one of these repositories has had to
discover the same things: that `actions/checkout` turns an annotated tag into a
lightweight one, that a bare `uv sync` in CI hides a lockfile someone forgot to
regenerate, that parallel branches appending to `CHANGELOG.md` conflict on every
squash merge. A fix worked out in one place stays there, and the next repository
pays for it again.

**Cross-repository work is friction.** panoptes/panoptes-pipeline#189 wants a
`uv` workspace spanning three of these. Three repositories that build, lint and
test the same way are a workspace; three that each do it differently are three
environments that happen to share a parent directory.

**Divergence is silent.** Two of the four scrape release notes out of
`CHANGELOG.md` with an `awk` pattern. A heading that does not match yields an
empty release body rather than an error, so the failure mode is a published
release that says nothing. Nobody notices, because nobody reads two
repositories' release workflows side by side.

The standard below is mostly not new work. It is what `panoptes-data` arrived at
after a run of small fixes, written down so the other three get it without
rediscovering it.

---

## 2. Build backend and versioning

**`hatchling` with `hatch-vcs`.** The nearest `v*` tag is the only source of
truth for the version, and no version is written in a file anywhere.

```toml
[build-system]
requires = ["hatchling>=1.25", "hatch-vcs>=0.4"]
build-backend = "hatchling.build"

[project]
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/panoptes"]
```

`setuptools` with `setuptools-scm` does the same job and was the previous answer
in `panoptes-utils` and POCS. The reason to converge on `hatchling` rather than
the reverse is narrow but real: `setuptools-scm`'s `version_file` writes a
`_version.py` into the source tree, which is then committed, gitignored or
excluded from lint depending on who set it up -- a generated artifact living in
`src/`, and one more thing each repository configures differently. `hatch-vcs`
writes nothing.

### 2.1 Migrating off `setuptools-scm`

The version file is the whole of the work. Anything importing `_version` moves
to:

```python
from importlib.metadata import version

__version__ = version("panoptes-utils")
```

Grep before deleting the file. A stale `from ._version import __version__` is an
`ImportError` when the package is imported, not a quiet degradation -- which is
the good case, but it will be the first thing that breaks.

### 2.2 A build needs the tags

Every CI job that builds the package or imports its version needs
`fetch-depth: 0` on `actions/checkout`. A default checkout fetches no tags, so
the version resolves to a development placeholder. This bites the docs build as
well as the release, because `mkdocstrings` imports the package.

---

## 3. Lint and format

**ruff, 110 columns, `select = ["E", "W", "F", "I", "UP"]`**, double quotes,
spaces, LF.

```toml
[tool.ruff]
line-length = 110
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"
```

Selecting the set explicitly is the point: ruff's default set moves between
versions, so an unpinned config makes "clean" mean something different on every
machine.

`UP` makes modern typing enforced rather than optional -- `tuple` not `Tuple`,
`X | None` not `Optional[X]`. Do not import from `typing` what the builtin
already provides.

Two rules that are about people rather than tools:

- **Do not hand-wrap code shorter than the limit.** The formatter joins it back
  and the diff is noise.
- **A format-only change goes in its own commit**, never mixed with a behavior
  change. Reviewing a real change through a reflow is how things get missed.

Repository-specific additions are fine where they say something. POCS's
`[tool.ruff.lint.isort]` section configuration, which groups `panoptes.utils`
and `panoptes.pocs` imports separately, is deliberate and stays.

---

## 4. Environments and dependencies

**`uv`, and nothing else.** No `hatch` environments, no `tox`, no
`requirements.txt`. `uv sync` installs the project plus the `dev` group, so the
package is importable and no `PYTHONPATH` is needed.

### 4.1 Groups, not extras

Tooling goes in `[dependency-groups]`. It never goes in
`[project.optional-dependencies]`.

```toml
[dependency-groups]
lint = ["ruff>=0.5.0"]
test = ["pytest", "pytest-cov", "pytest-doctestplus"]
docs = ["mkdocstrings[python]", "zensical"]
dev = [{ include-group = "lint" }, { include-group = "test" }, { include-group = "docs" }]
```

The distinction is not stylistic. An extra is published metadata: it is part of
what someone installing the package from PyPI can ask for, which is why
`panoptes-utils[images]` is right and `panoptes-utils[docs]` is not. A docs
toolchain is not something a user of the package installs.

Splitting `lint` from `test` also means the lint job installs ruff and not
pytest, and reports in well under a minute instead of waiting behind the test
environment.

### 4.2 The docs group does not relist runtime dependencies

`uv sync --group docs` installs the project too, so `mkdocstrings` can import
the package to read its docstrings. Listing the runtime dependencies again in
the docs group is how `panoptes-data`'s old `docs/requirements.txt` drifted
until it carried `photutils`, which that package does not import.

### 4.3 Standalone scripts declare their own dependencies

Anything in `scripts/` is [PEP 723](https://peps.python.org/pep-0723/), with an
inline metadata block, so it runs against a throwaway environment rather than
the synced one. A script's needs are not the package's needs.

---

## 5. Documentation

**[zensical](https://zensical.org), built from `zensical.toml` at the repository
root into `site/`.** Content lives in `docs/`.

### 5.1 The root Markdown files are not copied

`README.md`, `CHANGELOG.md`, `LICENSE.txt`, `CONTRIBUTING.md` and
`CODE_OF_CONDUCT.md` stay at the repository root, where GitHub and PyPI read
them. The pages under `docs/` include them from there:

```toml
pymdownx.snippets.base_path = ["."]
pymdownx.snippets.check_paths = true
```

`check_paths` matters: without it, a snippet pointing at a file that has moved
renders as nothing at all, and the page publishes with a hole in it.

This is also why there is no `edit_uri`. Nearly every page is a wrapper, so an
edit link would open the one-line wrapper rather than the file the reader is
actually looking at.

**A link in a root file must be absolute when its target is not published.**
`CONTRIBUTING.md` pointing at `plans/algorithm-design.md` works on GitHub and
404s on the site, because `plans/` is not in the nav. `--strict` does not catch
it: the build rewrites the path to a site URL and stays green, so the only
signal is a reader hitting the 404. Repository-relative links to `plans/`,
`LICENSE.txt`, `tests/` or anything else outside `docs/` are written as full
`https://github.com/...` URLs.

### 5.2 Docstrings are the API reference

A page in `docs/api/` is a `:::` block naming a module and nothing else. There
is no generated intermediate to commit and no second description of the API to
keep in step with the first.

```toml
"zensical.extensions.mkdocstrings".handlers.python.paths = ["src"]
"zensical.extensions.mkdocstrings".handlers.python.options.docstring_style = "google"
```

`paths = ["src"]` points griffe at the working tree, so the reference describes
this checkout rather than whichever copy of the package happens to be installed.

**Every package directory needs an `__init__.py`**, even where Python does not
require one. An implicit namespace package nested inside a regular package
imports perfectly well at runtime, and griffe -- a static reader -- cannot
follow it. `panoptes-pipeline`'s `utils` subpackage was in exactly that state,
and the reference would have omitted four modules without an error. Both
repositories that have built this site have hit it.

### 5.3 Working documents are not published

`plans/` is not in the nav. These are living documents full of provisional
measurements, several of which say so in their own text. Publishing them would
present working notes as documentation.

### 5.4 Publishing goes through the Pages deployment API

Not `mkdocs gh-deploy`, which force-pushes a built site to a `gh-pages` branch:
that keeps a second copy of the site in the repository's own history, and it
needs `contents: write` on a job that executes branch code.

The replacement splits the work in two, and the permissions follow the split.
The build job runs on pull requests, executes the branch's code -- importing the
package is how `mkdocstrings` works -- and gets `contents: read` and nothing
else. Only the deploy job, which runs from `main` and builds nothing, can mint
an OIDC token or call the Pages API.

Requires Settings -> Pages -> Source set to **GitHub Actions**, once, per
repository.

---

## 6. Continuous integration

Four workflows. Each answers a different question.

### 6.1 `tests.yml` -- is this change good?

Separate `lint` and `test` jobs, so "which one failed" does not need the log
opened. Lint runs `ruff check` and `ruff format --check` as separate steps, for
the same reason.

```yaml
- run: uv sync --locked --group test --python ${{ matrix.python-version }}
```

**`--locked` is not optional.** It fails rather than re-resolving, so a
dependency edit that skipped `uv lock` is caught here instead of silently
testing something other than what the lockfile describes.

The matrix is every Python version `requires-python` claims. A package
declaring `>=3.12` and testing only 3.12 is claiming support it does not
measure.

Coverage is configured in `pyproject.toml`, uploaded as an artifact, and sent to
Codecov with `fail_ci_if_error: false` -- a coverage upload failing is not a
reason to fail a test run that passed.

### 6.2 `docs.yml` -- does the site still build?

Builds on pull requests with `--strict`, so a link to a page that does not exist
or a `:::` block naming an unimportable module fails the build rather than
publishing a hole. Uploads the artifact on pull requests too, so a reviewer can
download the built site. Deploys only from `main`.

### 6.3 `create-release.yml` -- see section 7.

### 6.4 `canary.yml` -- has something upstream broken us?

The committed `uv.lock` makes pull-request CI attributable: a red run means the
change under review broke something, not that a dependency shipped overnight.
The cost is that nothing then notices when an upstream release *does* break us
-- the pinned tree keeps passing until someone relocks, which may be weeks later
and far from the cause.

The canary is the other half. Weekly, plus `workflow_dispatch`, it ignores the
lockfile (`uv sync --upgrade`), takes the newest versions the constraints allow,
prints the resolved tree and runs the suite. Nothing is committed: the run is a
measurement, and the lockfile changes only when a human relocks deliberately.

A failure here is a question, not necessarily a bug. The answer is either a code
change or an upper bound in `pyproject.toml`.

### 6.5 `uv run` syncs unless told not to

A step that runs `uv run pytest` after an install step re-syncs first, against
the lockfile and the default groups. That silently undoes whatever the install
step chose, and the two places it matters are both places where the choice was
the point:

- After `uv sync --no-default-groups --group test`, a bare `uv run` pulls the
  default `dev` group back in.
- In the canary, after `uv sync --upgrade`, a bare `uv run` **restores the
  pinned tree** and tests it -- a canary that cannot fail.

So every `uv run` in CI that follows an install step passes `--no-sync`.

### 6.6 Things not to do

- **No `actions/setup-python`.** `uv` resolves an interpreter matching
  `requires-python` on its own.
- **Pin action majors and keep them current.** `astral-sh/setup-uv@v6`,
  `actions/checkout@v5`.
- **Permissions per job, not workflow-wide.** A job that executes branch code
  gets `contents: read`.

---

## 7. Releases

**The annotated tag message is the release notes.** One place to write them, and
the tag and the release cannot drift apart.

The alternative -- and the previous answer in `panoptes-utils` and POCS -- is to
scrape the notes out of `CHANGELOG.md` with an `awk` range. It fails quietly:
a heading the pattern does not match produces an empty release body rather than
an error.

The workflow triggers on `v[0-9]+.[0-9]+.[0-9]+` and, in order:

1. Checks out with `fetch-depth: 0`, because `hatch-vcs` needs the history.
2. **Restores the annotated tag object.** `actions/checkout` fetches the commit
   SHA into the tag ref, so the checkout holds a *lightweight* tag whatever the
   remote has. Without this the next step fails on every release.
3. **Requires the tag to be annotated, and extracts the notes there.**
   `%(contents)` on a lightweight tag reports the *commit* message instead,
   which would publish a commit subject as the release notes -- wrong, and
   plausible enough to go unnoticed. Both checks -- annotated, and a message
   with content in it -- run before the build, and the notes are written to
   `$RUNNER_TEMP` so the build still sees a clean tree. Doing the
   empty-message check at the end instead is the trap: the upload succeeds, the
   release step fails, and PyPI refuses a re-upload of that version, so there
   is nothing to retry.
4. `uv build`, which drives whatever backend `[build-system]` declares, so the
   artifacts are the ones `uv build` produces locally.
5. Publishes with **Trusted Publishing** -- an OIDC token minted for the
   workflow, not a long-lived API token in a secret. No user or password on the
   publish step: an explicit password disables Trusted Publishing and with it
   the build attestations the action produces by default.
6. Creates the GitHub release, keyed on the **build** succeeding rather than the
   publish. A release needs the tag and the dists, not a successful upload. It
   edits an existing release rather than failing on one, so a re-run is safe.

Trusted Publishing needs a one-time setup on PyPI per project: Manage project ->
Publishing -> Add a new publisher, naming owner, repository and workflow
filename. For a package not yet on PyPI it is a *pending* publisher, created the
same way from the account's publishing page. Without it the first tag builds and
then fails at upload.

### 7.1 Versioning

[Semantic versioning](https://semver.org), with the tag as the only source of
truth. Every one of these repositories is pre-1.0 deliberately: at `0.y.z` the
public API is not stable, which is an accurate description of packages whose
modules are still being deleted and whose CLIs are being rewritten.

While pre-1.0, **minor** covers new capability or a breaking change -- both,
because at `0.x` there is no separate channel for breaks -- and **patch** covers
fixes and internal work that changes no interface.

Tag on `main` only, annotated, with a message saying what the release contains.
`archive/*` tags are not releases; they preserve retired branch tips and carry
no version meaning.

### 7.2 The changelog moves with the branch

`CHANGELOG.md` is updated in the branch that makes the change, not afterwards. A
branch is not finished until its entry is under `## Unreleased`.
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) headings, only the ones
that apply. Releasing means renaming `## Unreleased` to `## vX.Y.Z -- YYYY-MM-DD`;
that section and the annotated tag message say the same thing, so write it once
and reuse it.

---

## 8. Repository files

### 8.1 `.gitattributes`

```
CHANGELOG.md merge=union
```

Every branch appends under `## Unreleased`, at the same place, and `main` moves
between the branch starting and merging. Squash merges make it worse: the merged
commit shares no ancestry with the branch's own, so git sees two unrelated edits
to the same lines and stops. The resolution is always "keep both sides", which
is exactly what `merge=union` does.

Read the result before pushing. Union takes both sides' lines in order and never
reports a conflict, so bullets can land in either order and a blank line can be
dropped. Wrong order is possible; lost content is not.

### 8.2 `.pre-commit-config.yaml`

The same tools CI runs, at the same settings, so a hook cannot disagree with a
job: `ruff-check --fix` and `ruff-format`, both reading `pyproject.toml`, plus
the cheap `pre-commit-hooks` checks for trailing whitespace, end-of-file, merge
conflict markers and TOML/YAML/JSON syntax.

**Binary fixtures are excluded from the text hooks, per hook.**
`trailing-whitespace`, `end-of-file-fixer` and `mixed-line-ending` will happily
"fix" a binary file: run once over `panoptes-pipeline`, all three rewrote bytes
inside `tests/data/solved.fits.fz` and `tests/data/widefield.fits.fz` and
reported it as a fix. `panoptes-data` commits no binaries and so has not met
this.

The exclusion goes on those three hooks and not at the top level. A top-level
`exclude` applies to *every* hook in the file, which would also take the binary
fixtures out of `check-added-large-files` and `check-merge-conflict` -- the two
that most need to see them, and the ones the exclusion is meant to leave alone.

**The `ruff-pre-commit` rev equals the `ruff` version in `uv.lock`.** Otherwise
the hook and the CI lint job are different binaries, and a formatting change
between two patch releases means the hook rewrites what CI then rejects, or the
reverse. When the lockfile's ruff moves, the rev moves with it.

Contributors run `pre-commit install` once. CI remains the actual gate -- a hook
is a convenience, not an enforcement mechanism, because anyone can pass `-n`.

### 8.3 Community files

`AUTHORS.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE.txt`,
`README.md`, `CHANGELOG.md`. `CONTRIBUTING.md` describes the repository's actual
workflow rather than a generic one -- if it says "run the tests" without saying
`uv run pytest`, it is decoration.

### 8.4 Agent instructions

**`AGENTS.md` is the real file. `CLAUDE.md` and `GEMINI.md` are symlinks to
it.**

```bash
git mv CLAUDE.md AGENTS.md
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md GEMINI.md
```

Three tools read three filenames, and where more than one file has existed they
have been copies -- which is to say, drifting copies. Git stores a symlink as a
blob holding the target path, so this survives clone and checkout.

The caveat, noted so the choice is deliberate: git on Windows without
`core.symlinks` checks a symlink out as a text file containing the path. Nobody
develops these repositories on Windows today.

---

## 9. Branches, issues and cross-repository work

`main` is the default and only long-lived branch. Feature branches are cut from
it and merged back; retired tips are preserved as `archive/*` tags rather than
kept as branches.

Branch names say what the branch is for: `type/issue-NNN` plus an optional short
description, as in `cleanup/issue-170-tooling-foundation`. A generated name
carrying neither the type nor the issue number is not one; rename it before
pushing.

**`plans/` is the reasoning; GitHub issues are the state.** Why a thing is worth
doing belongs in a plan, whether it is planned or done belongs in the tracker,
and never both.

**Work is filed where the code lives.** A POCS change is a POCS issue, even when
the reasoning for it is in this repository's `plans/`. The "Photometry rebuild"
project board spans all four and is the single view across them. Cross-repository
references are fully qualified everywhere -- `panoptes/POCS#1410`, never a bare
`#1410` -- including in conversation, where a bare number is read against
whichever repository is in front of the reader.

---

## 10. Where each repository stands

Conformance is tracked in the issues, not here. This section says only where
each repository started, so the migration issues have something to point at, and
where the sections above came from.

**This repository** now implements all of them. The three sections it
contributed back, each from a failure met while migrating, are 5.2 (a package
directory without `__init__.py` is invisible to griffe), 6.5 (`uv run` re-syncs
and silently undoes the install step) and 8.2 (the whitespace hooks corrupt
binary fixtures).

**`panoptes-data`** is the reference for 2, 4, and 5 through 8, which it arrived
at over a run of small fixes. It should pick up 3 (it lints at 100 columns
without `W`), the 6.5 `--no-sync` fix -- its canary has the bug described there
-- and the 6.1 matrix, since it tests only 3.12 while claiming `>=3.12`.

**`panoptes-utils` and POCS** are the reference for 3, which is why 3 is what it
is. Both started on `setuptools-scm`, mkdocs-material with `gh-deploy`, and
changelog-scraped release notes, so 2, 5, 6 and 7 are the work in each.
