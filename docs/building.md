# Building the docs

The site is built with [Zensical](https://zensical.org/) from `zensical.toml`
at the repository root and the Markdown under `docs/`.

```bash
uv sync --group docs
uv run --group docs zensical serve     # live reload at http://127.0.0.1:8000
uv run --group docs zensical build --clean --strict
```

`--strict` turns warnings into errors. A link to a page that does not exist, or
a `:::` block naming a module that cannot be imported, fails the build rather
than publishing a hole. CI builds this way on every pull request, so a page that
only works locally will not merge.

## How the pages are put together

There are two kinds of page here and no third.

**Wrappers.** `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `AUTHORS.md` and
`LICENSE.txt` live at the repository root, where GitHub and PyPI read them. The
pages under `docs/` include them from there with a snippet line:

```markdown
--8<-- "README.md"
```

`pymdownx.snippets.base_path` is the repository root and `check_paths` is on, so
a snippet pointing at a file that has moved fails the build. Without that check
the page renders as nothing at all and publishes with a hole in it.

This is also why there is no `edit_uri` in the configuration. Nearly every page
is a one-line wrapper, so an edit link would open the wrapper rather than the
file the reader is actually looking at.

**API reference.** A page under `docs/api/` is a `:::` block naming a module,
and nothing else:

```markdown
# `panoptes.pipeline.lightcurve.core`

::: panoptes.pipeline.lightcurve.core
```

`mkdocstrings` reads the docstrings; there is no generated intermediate to
commit and no second description of the API to keep in step with the first. A
new module means a new page here and a line in the `nav`. Improving a docstring
improves the reference, which is the point.

`handlers.python.paths = ["src"]` points griffe at the working tree, so the
reference describes this checkout rather than whichever copy of the package
happens to be installed.

## What is not published

`plans/` is not in the nav, deliberately. Those are living documents full of
provisional measurements -- several say so in their own text -- and publishing
them would present working notes as documentation. They are read in the
repository, where their status is obvious.

## Publishing

`docs.yml` builds the site on every pull request and uploads it as an artifact,
so a reviewer can download it. On `main` a second job deploys that artifact
through the GitHub Pages API. Nothing is force-pushed to a `gh-pages` branch,
and the job that runs branch code has no write permission of any kind.
