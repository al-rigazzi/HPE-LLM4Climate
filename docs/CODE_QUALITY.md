# Code Quality Tools Installation Guide

This document explains how to set up the optional code quality tools for the HPE-LLM4Climate project.

## Quick Installation

Install all code quality tools at once:

```bash
pip install black isort pylint mypy
```

Or install from requirements.txt (which includes these tools):

```bash
pip install -r requirements.txt
```

## Individual Tool Installation

### Black (Code Formatter)
```bash
pip install black>=24.0.0
```

### isort (Import Sorter)
```bash
pip install isort>=5.12.0
```

### pylint (Code Linter)
```bash
pip install pylint>=3.0.0
```

### mypy (Type Checker)
```bash
pip install mypy>=1.8.0
```

## Pre-commit Hooks

The project includes pre-commit hooks that will automatically run code formatting tools (if available) before each commit.

**Currently enabled in pre-commit:**
- ✅ `black` - Code formatting (optional - runs if available)
- ✅ `isort` - Import sorting (optional - runs if available)
- ❌ `mypy` - Type checking (disabled in pre-commit hooks)
- ❌ `pylint` - Code linting (disabled in pre-commit hooks)

### Install pre-commit
```bash
pip install pre-commit
```

### Set up pre-commit hooks
```bash
pre-commit install
```

### Manual pre-commit run
```bash
pre-commit run --all-files
```

**Note:** mypy and pylint are available for manual use but are not run automatically during commits.

## Manual Code Quality Checks

You can also run the code quality checks manually using the provided script:

```bash
./scripts/code-quality-check.sh
```

This script will:
- ✅ Run each tool if available
- ⏭️ Skip tools that are not installed (no failure)
- 📁 Only process Python files in the `multimodal_aifs/` directory
- 🔍 Includes mypy and pylint (not run by pre-commit hooks)

## Tool Configuration

### Black Configuration
- Line length: 88 characters
- Profile: Compatible with isort

### isort Configuration
- Profile: black (for compatibility)
- Line length: 88 characters

### mypy Configuration
- Uses project's `mypy.ini` configuration
- No error summary output
- Shows error codes
- **Note:** Currently disabled in pre-commit hooks - run manually when needed

### pylint Configuration
- No scoring output
- No reports (warnings/errors only)
- **Note:** Disabled in pre-commit hooks - run manually when needed

## Pre-commit Behavior

The pre-commit hooks currently focus on code formatting and basic checks:

**Automatic (on every commit):**
- Standard checks (trailing whitespace, YAML validation, etc.)
- Code formatting with `black` (if available)
- Import sorting with `isort` (if available)

**Manual only:**
- Type checking with `mypy` - Run with: `mypy multimodal_aifs`
- Code linting with `pylint` - Run with: `pylint multimodal_aifs`

## Graceful Degradation

The pre-commit hooks are designed to work even if some tools are missing:

- If `black` is not available: Skips code formatting
- If `isort` is not available: Skips import sorting
- `mypy` and `pylint` are disabled from pre-commit hooks, so they don't affect commits

This ensures that contributors can work on the project even without all tools installed, while encouraging best practices when tools are available.
