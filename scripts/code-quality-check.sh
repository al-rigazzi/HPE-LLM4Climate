#!/bin/bash
# Pre-commit helper script for optional tool execution
# This script runs code quality tools if available, gracefully handling missing tools

set -e  # Exit on error, but individual tool failures are handled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

echo "🔍 Running optional code quality checks..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run tool if available
run_if_available() {
    local tool="$1"
    shift
    
    if command_exists "$tool"; then
        echo "✅ Running $tool..."
        "$tool" "$@" || {
            echo "⚠️  $tool found issues but continuing..."
            return 0  # Don't fail the entire script
        }
    else
        echo "⏭️  $tool not found - skipping"
    fi
}

# Change to project root
cd "$PROJECT_ROOT"

# Find Python files in multimodal_aifs
PYTHON_FILES=$(find multimodal_aifs -name "*.py" -type f 2>/dev/null || echo "")

if [ -z "$PYTHON_FILES" ]; then
    echo "📁 No Python files found in multimodal_aifs/"
    exit 0
fi

echo "📂 Found Python files in multimodal_aifs/"

# Run isort if available
if [ -n "$PYTHON_FILES" ]; then
    run_if_available isort --profile black --line-length 88 $PYTHON_FILES
fi

# Run black if available
if [ -n "$PYTHON_FILES" ]; then
    run_if_available black --line-length 88 $PYTHON_FILES
fi

# Run mypy if available
run_if_available mypy multimodal_aifs --no-error-summary --show-error-codes

# Run pylint if available
run_if_available pylint multimodal_aifs --score=no --reports=no

echo "✨ Code quality checks completed!"
