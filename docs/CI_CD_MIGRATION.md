# CI/CD Configuration (Current State)

This repository currently maintains CI pipelines on both GitHub and GitLab.

## Active Pipeline Files

- GitHub Actions: `.github/workflows/ci.yml`
- GitLab CI: `.gitlab-ci.yml`

## Current Pipeline Scope

### Test jobs

- Python 3.12
- Unit tests on Linux and macOS (GitHub)
- Targeted integration subset on Linux runners
- Integration coverage on macOS with `-m "not large_memory"`

### Lint and quality jobs

- `black --check`
- `isort --check-only`
- `flake8` (informational)
- `pylint` (informational)
- `mypy` (GitLab, informational)

### Security jobs

- `safety`
- `bandit`

## Shared CI Environment Behavior

Both pipelines configure these runtime flags for deterministic CI:

- `USE_MOCK_LLM=true`
- `USE_QUANTIZATION=false`
- `USE_REAL_ZARR=false`
- `ZARR_V3_EXPERIMENTAL_API=1`

## Sync Guidance

When changing test, lint, or dependency behavior:

1. Update `.github/workflows/ci.yml`
2. Mirror equivalent behavior in `.gitlab-ci.yml`
3. Validate both pipelines before merge

## Notes

- GitLab macOS job remains optional and commented in `.gitlab-ci.yml`
- Linux jobs use CPU-only PyTorch wheels where configured
