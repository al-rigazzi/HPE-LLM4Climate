# Dual CI/CD Configuration: GitHub Actions + GitLab CI/CD

This repository is configured to run CI/CD pipelines on both GitHub and GitLab platforms. This ensures your code quality and testing standards are maintained regardless of which platform hosts the repository.

## Overview

### GitHub Actions (`.github/workflows/ci.yml`)
- **Platforms**: Ubuntu (latest), macOS (latest)
- **Python Version**: 3.12
- **Jobs**:
  - `test`: Unit and integration tests on both OS platforms
  - `lint`: Code quality checks (Black, isort, Flake8, Pylint)
  - `security`: Security scanning (Safety, Bandit)

### GitLab CI/CD (`.gitlab-ci.yml`)
- **Platforms**: Ubuntu (Python 3.12 image)
- **Optional**: macOS support (requires GitLab Runner with macOS executor)
- **Stages**:
  - `test`: Unit and integration tests
  - `lint`: Code quality checks (Black, isort, Flake8, Pylint, mypy)
  - `security`: Security scanning (Safety, Bandit)

## Configuration Details

### GitHub Actions Setup
GitHub Actions are automatically enabled when you push to the repository. No additional setup is required. The workflow triggers on:
- Pull requests to the `main` branch
- Uses built-in concurrency control to cancel stale runs

### GitLab CI/CD Setup

#### Basic Setup (Linux Runners)
GitLab CI/CD will automatically run on your GitLab instance if you have:
1. **Shared Runners** enabled (default for gitlab.com)
2. **Project CI/CD enabled** in project settings (default)

No additional configuration needed for Ubuntu-based tests.

#### Optional: macOS Testing on GitLab
If you want to test on macOS in GitLab, you need a dedicated macOS Runner:

1. **Set up a GitLab Runner on macOS**:
   ```bash
   # Install GitLab Runner on macOS
   brew install gitlab-runner

   # Register the runner
   gitlab-runner register \
     --url https://gitlab.com/ \
     --registration-token <your-token> \
     --executor shell \
     --tag-list macos

   # Start the runner
   gitlab-runner install
   gitlab-runner start
   ```

2. **Uncomment the macOS job** in `.gitlab-ci.yml`:
   - Find the commented section at the bottom of the file
   - Uncomment the `test:unit:macos` job
   - The job will use the `macos` tag to route to your macOS runner

## Pipeline Features

### Caching
Both systems cache pip dependencies for faster builds:
- **GitHub Actions**: `.cache/pip` with fallback keys per OS
- **GitLab CI/CD**: `.cache/pip` with automatic restoration

### Artifacts
Test results and security reports are saved:
- **GitHub Actions**: Uploaded as artifacts with 30-day retention
- **GitLab CI/CD**: Available in Artifacts section with configurable retention (90 days for security, 30 days for tests)

### Error Handling
- **Hard failures**: Black, isort (must pass)
- **Soft failures** (allow_failure: true):
  - Flake8 (informational)
  - Pylint (informational)
  - mypy (optional type checking)
  - Safety (informational)
  - Bandit (informational)

### Environment Variables
Both systems set the same environment variables for consistent testing:
- `USE_MOCK_LLM=true` - Uses mock implementations during testing
- `USE_QUANTIZATION=false` - Disables quantization in tests
- `USE_REAL_ZARR=false` - Uses test datasets
- `ZARR_V3_EXPERIMENTAL_API=1` - Enables zarr experimental features
- `PYTORCH_*` - PyTorch configuration for CPU testing

## Migration Instructions

### Step 1: Verify GitHub Actions (Already Done)
Your existing `.github/workflows/ci.yml` is preserved and remains active.

### Step 2: Set Up GitLab Repository
1. Create/import the repository into GitLab
2. Push all branches including the `.gitlab-ci.yml` file
3. Verify CI/CD is enabled in **Settings > CI/CD > General pipelines**

### Step 3: Configure GitLab Runners (Optional)
If you need custom runners or macOS support:
1. Go to **Settings > CI/CD > Runners**
2. Follow the setup instructions in the "macOS Testing" section above
3. Uncomment the macOS job in `.gitlab-ci.yml`

### Step 4: Update Repository Links
Update any documentation or CI badges in your README:

**GitHub Actions badge**:
```markdown
[![CI/CD Pipeline](https://github.com/al-rigazzi/HPE-LLM4Climate/actions/workflows/ci.yml/badge.svg)](https://github.com/al-rigazzi/HPE-LLM4Climate/actions)
```

**GitLab CI/CD badge** (add after migration):
```markdown
[![pipeline status](https://gitlab.com/al-rigazzi/HPE-LLM4Climate/badges/main/pipeline.svg)](https://gitlab.com/al-rigazzi/HPE-LLM4Climate/-/commits/main)
```

## Key Differences Between GitHub and GitLab CI/CD

| Feature | GitHub Actions | GitLab CI/CD |
|---------|---|---|
| **Trigger** | PR/push events | Any ref update |
| **Concurrency Control** | Built-in `concurrency` block | Manual configuration |
| **Matrix Strategy** | Native `strategy.matrix` | `parallel:matrix` |
| **Soft Failures** | `continue-on-error: true` | `allow_failure: true` |
| **Docker Images** | Uses `runs-on` (pre-configured) | Uses `image:` (Docker image URI) |
| **Artifacts** | Auto-upload after job | Explicit `artifacts:` section |
| **Caching** | Key-based with restore keys | Path-based caching |
| **OS Support** | Ubuntu, macOS, Windows | Any Docker image + custom runners |

## Maintenance Notes

### When to Update
Keep both workflows synchronized when:
- Adding/removing test suites
- Changing Python version requirements
- Adding new linting or security tools
- Modifying environment configuration

### Recommended Approach
1. Update the GitHub Actions workflow first (testing locally)
2. Mirror changes to `.gitlab-ci.yml`
3. Test both pipelines before merging to main

### Debugging CI Failures

**GitHub Actions**:
- Check the "Actions" tab in the repository
- View detailed logs for each step
- Use local `act` tool: `act -l` to list jobs, `act -j <job-name>` to run locally

**GitLab CI/CD**:
- Check the "CI/CD > Pipelines" section
- View pipeline graphs and job logs
- Download artifacts from the pipeline details page
- Use GitLab Runner debug: `gitlab-runner verify`

## Security Scanning Results

Both systems generate security scan results:

### GitHub Actions
- Artifacts stored in Actions tab
- `safety-results.json` - Python package vulnerability report
- `bandit-results.json` - Code security analysis

### GitLab CI/CD
- Artifacts in pipeline details
- Same JSON format for consistency
- Can be integrated with GitLab Security Dashboard

## Questions & Support

For platform-specific issues:
- **GitHub Actions**: Consult [GitHub Actions Documentation](https://docs.github.com/en/actions)
- **GitLab CI/CD**: Consult [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)

For test/linting tool documentation:
- **pytest**: https://docs.pytest.org
- **Black**: https://black.readthedocs.io
- **Pylint**: https://www.pylint.org
- **mypy**: https://www.mypy-lang.org
