# Pre-commit Hook Documentation

## Overview
This repository includes a pre-commit hook that automatically runs code formatting tools on staged Python files before each commit.

## Features
- **Automatic Import Sorting**: Runs `isort` to sort and organize Python imports
- **Code Formatting**: Runs `black` to format Python code according to PEP 8 standards
- **Graceful Degradation**: Only runs tools that are available in the environment
- **Non-blocking**: If tools are not installed, the hook provides helpful installation instructions but doesn't block commits

## What It Does
1. Detects staged Python files (`.py` files)
2. Runs `isort` if available to sort imports
3. Runs `black` if available to format code
4. Re-stages any modified files
5. Provides helpful feedback and installation tips

## Installation Instructions
To get the most out of this hook, install the recommended tools:

```bash
# Using pip
pip install isort black

# Using conda
conda install -c conda-forge isort black
```

## Hook Behavior
- ✅ **Runs automatically** on every `git commit`
- ✅ **Non-destructive** - only formats staged files
- ✅ **Helpful messages** - tells you what it's doing
- ✅ **Installation tips** - suggests how to install missing tools
- ✅ **Fast** - only processes staged Python files

## Example Output
```
🔧 Running pre-commit hooks...
📁 Found staged Python files:
multimodal_aifs/core/aifs_climate_fusion.py

🔄 Running isort...
✅ isort: No changes needed

🔄 Running black...
✅ black: No changes needed

🔄 Re-staging potentially modified files...
✅ Files re-staged

🎉 Pre-commit hooks completed successfully!
```

## Customization
The hook is located at `.git/hooks/pre-commit`. You can modify it to:
- Add additional formatting tools
- Change formatting options
- Add custom checks
- Skip certain file patterns

## Benefits
- **Consistent Code Style**: All committed code follows the same formatting standards
- **Automatic**: No need to remember to run formatters manually
- **Team Consistency**: Ensures all team members' code follows the same style
- **CI/CD Ready**: Code is already properly formatted when pushed to CI/CD pipelines