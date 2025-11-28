#!/usr/bin/env python3
"""
Script to add Apache 2.0 license header to all Python files in the repository.

Usage:
    python scripts/add_license_headers.py [--dry-run]
"""

import argparse
from pathlib import Path

# Standard Apache 2.0 header for Python files
LICENSE_HEADER = """# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""


def has_license_header(content: str) -> bool:
    """Check if file already has a license header."""
    return "Licensed under the Apache License" in content or "Copyright 2025" in content


def add_header_to_file(filepath: Path, dry_run: bool = False) -> bool:
    """Add license header to a Python file if it doesn't have one."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Skip if already has license header
        if has_license_header(content):
            return False

        # Handle shebang and encoding declarations
        lines = content.split("\n")
        insert_pos = 0
        new_content = []

        # Preserve shebang
        if lines and lines[0].startswith("#!"):
            new_content.append(lines[0])
            insert_pos = 1

        # Preserve encoding declaration
        if len(lines) > insert_pos and "coding" in lines[insert_pos]:
            new_content.append(lines[insert_pos])
            insert_pos += 1

        # Add license header
        new_content.append(LICENSE_HEADER.rstrip())

        # Add original content (skip preserved lines)
        remaining_content = "\n".join(lines[insert_pos:])
        if remaining_content.strip():
            new_content.append(remaining_content)

        final_content = "\n".join(new_content)

        if not dry_run:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_content)
            print(f"✓ Added header to: {filepath}")
        else:
            print(f"[DRY RUN] Would add header to: {filepath}")

        return True

    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False


def find_python_files(root_dir: Path) -> list[Path]:
    """Find all Python files in the repository."""
    python_files = []

    # Directories to skip
    skip_dirs = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "venv",
        "env",
        ".venv",
        "node_modules",
        ".vscode",
        ".idea",
        "build",
        "dist",
        "*.egg-info",
        "aifs-single-1.1",  # Submodule - don't modify
    }

    for py_file in root_dir.rglob("*.py"):
        # Skip files in excluded directories
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue

        python_files.append(py_file)

    return sorted(python_files)


def main():
    """Main function to process all Python files."""
    parser = argparse.ArgumentParser(description="Add Apache 2.0 license headers to Python files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent.parent

    print("🔍 Finding Python files...")
    python_files = find_python_files(repo_root)
    print(f"Found {len(python_files)} Python files")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    # Process each file
    modified_count = 0
    skipped_count = 0

    for filepath in python_files:
        if add_header_to_file(filepath, dry_run=args.dry_run):
            modified_count += 1
        else:
            skipped_count += 1

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY RUN] Would modify: {modified_count} files")
        print(f"[DRY RUN] Would skip: {skipped_count} files (already have headers)")
    else:
        print(f"✓ Modified: {modified_count} files")
        print(f"⊘ Skipped: {skipped_count} files (already have headers)")
    print("=" * 60)


if __name__ == "__main__":
    main()
