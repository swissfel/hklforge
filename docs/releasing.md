# Releasing (maintainers)

hklforge is released the same way as escape-fel: the version comes from a git
tag, and pushing `main` releases.

## Versions come from git tags

The version is not written in the source. setuptools-scm reads it from the
git tag: tag `v0.1.0` gives version `0.1.0`, and a commit after a tag gets a
`.devN+g<hash>` version. `hklforge.__version__` shows the installed version.
To release, create the right tag; never put a version into `pyproject.toml`.

## Pushing `main` releases (`.githooks/pre-push`)

Opt in once per clone:

```bash
git config core.hooksPath .githooks
```

Then every push of `main` to `origin` creates the next PATCH tag
(`v0.1.0` → `v0.1.1`) and pushes it too. That tag starts
`.github/workflows/publish.yml`:

1. the full test suite and the core-dependency jobs (`ci.yml`)
2. build the sdist and wheel, then `twine check --strict`
3. check that the built version equals the tag, and that the sdist contains
   `LICENSE`, `spacegroups.json` and the geometry YAML files
4. upload to PyPI by trusted publishing (no token)

What the hook does in other cases:

- **MINOR / MAJOR releases** (a new feature or a breaking change): tag the
  commit yourself before pushing (`git tag v0.2.0`). The hook sees the commit
  is already tagged and adds nothing.
- **A push that should not release**: `SKIP_AUTOTAG=1 git push`.
- **Other branches** are never tagged.

> **Note:** the tag before 0.1.0 is `v0.0.1`. A plain push would therefore
> release **0.0.2**. For the 0.1.0 release, run `git tag v0.1.0` before pushing.

The PyPI trusted publisher is registered for this repository, the workflow
file **`publish.yml`** and the GitHub environment **`pypi`**. If either is
renamed, the upload fails.

## Dependencies

The hard dependencies are `numpy>=1.24` and `pyyaml>=6.0`. These floors are
the oldest versions the suite is run against: the CI `core` job on
Python 3.10. Everything else is an extra (`crystal`, `draw`, `test`,
`examples`).

Keep it that way:

- `import hklforge` and every module except `hklforge.crystal.reflections`
  (pandas) must import with numpy and pyyaml alone. The `core` job checks
  this, and the conda-forge package test relies on it.
- An optional package is imported inside the function or module that needs
  it, never at the top of a module that is imported anyway.
- Tests that need an optional package use `pytest.importorskip`.

## conda-forge

hklforge is not on conda-forge yet. The recipe for the submission is
[`dev/conda-forge-recipe.yaml`](../dev/conda-forge-recipe.yaml) (v1 format).
It was built and tested locally with `rattler-build` and passes
`conda-smithy recipe-lint`. The test runs on Python 3.11 (conda-forge's
`python_min`) and 3.15, with `pip check`.

To submit, after the version is on PyPI:

1. Put the PyPI sdist's sha256 into the recipe
   (`curl -sL https://pypi.org/packages/source/h/hklforge/hklforge-X.Y.Z.tar.gz | sha256sum`)
   and check that `version` matches.
2. Fork `conda-forge/staged-recipes` and add the recipe as `recipes/hklforge/recipe.yaml`.
3. Open the PR and comment `@conda-forge/help-python, ready for review!`.
4. Watch the PR for **inline review comments** on the diff and for "Changes
   requested". They are easy to miss next to the conversation tab.

After the PR is merged, the recipe lives in `conda-forge/hklforge-feedstock`:

- Enable the bot's automerge with `bot: automerge: true` in its
  `conda-forge.yml`, via a PR from a fork.
- Delete `dev/conda-forge-recipe.yaml` here, so no stale copy remains.

From then on, each PyPI release produces a bot PR on the feedstock, which
merges itself when green. The bot does **not** update dependencies. A change
to `[project] dependencies` must also go into the feedstock's
`requirements: run:` by hand.

Lessons from escape-fel's submission, already applied to this recipe:

| pitfall | here |
|---|---|
| the recipe pointed at a version that was never uploaded (the source URL returned 404) | step 1: sha256 from the actual PyPI file |
| an unconditional import of a package that is not on conda-forge | only numpy and pyyaml at import (CI `core` job) |
| unpinned dependencies let the solver pick an ancient release on the newest Python | floors on every run dependency |
| `python >=3.9` instead of the `python_min` pins | `python ${{ python_min }}.*` / `>=${{ python_min }}` |
| test on one Python only | `python_version: [${{ python_min }}.*, "*"]` |
| `pip_check: false` without a reason | pip check on (default) |
| old meta.yaml format | v1 `recipe.yaml` |
| no "ready for review" comment, overlooked inline comments | steps 3 and 4 |
