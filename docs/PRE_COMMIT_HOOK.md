# Pre-commit Hooks (Current Setup)

This repository uses `.pre-commit-config.yaml` with standard hooks plus optional local formatting hooks.

## Install

```bash
pip install pre-commit
pre-commit install
```

## What Runs on Commit

From `pre-commit-hooks`:

- trailing whitespace cleanup
- end-of-file fixer
- YAML validation
- large file checks
- merge conflict checks
- debug statement checks

From local hooks (for `multimodal_aifs/*.py`):

- optional `isort` formatting (runs only if command is available)
- optional `black` formatting (runs only if command is available)

## Manual Hook Stages

The repository also includes manual-stage fallback hooks for official `isort` and `black` repos.

Run all hooks manually:

```bash
pre-commit run --all-files
```

Run manual-stage formatters explicitly:

```bash
pre-commit run isort --hook-stage manual --all-files
pre-commit run black --hook-stage manual --all-files
```

## Notes

- Hook scope is intentionally focused on `multimodal_aifs/*.py`
- CI still enforces formatting/linting checks independently
