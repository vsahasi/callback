# CI workflow, pending a token scope

`github-actions-ci.yml` is the GitHub Actions workflow for this repo. It
lives here rather than at `.github/workflows/ci.yml` for one reason: the
token that pushed this branch has `repo` scope but not `workflow`, and
GitHub rejects any push whose commits add or change a file under
`.github/workflows/`. Rather than drop the file, it is parked here.

## Enabling it

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git rm ci/README.md
git commit -m "ci: enable the GitHub Actions workflow"
git push
```

## What it does

Three jobs on every push and pull request:

- **test**: `uv sync --locked`, then the full suite.
- **lint**: `ruff check` with GitHub annotations, and `ruff format --check`.
- **modal-preflight**: `scripts/modal_preflight.py`, which validates a
  deploy without performing one and needs no secrets.

The snapshot step runs first and alone, so contract drift is the first red
step. It pins `CALLBACK_UPDATE_SNAPSHOT` empty and asserts `schemas/` is
unchanged, because setting that variable to `1` makes the drift test skip
and pass while rewriting the committed snapshot.

It is validated with `actionlint` (0 findings), so it is ready to run as
soon as it is moved into place.
