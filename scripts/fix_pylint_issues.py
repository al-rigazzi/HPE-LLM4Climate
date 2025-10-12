# Copyright 2025 Hewlett Packard Enterprise Development LP
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
"""Quick script to batch fix common pylint issues."""

import re
import subprocess
from pathlib import Path


def fix_f_strings_in_file(filepath):
    """Remove f prefix from strings without interpolation."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    original = content
    # Pattern for f"..." or f'...' without {
    content = re.sub(r'f"([^"]*?)"(?![^"]*\{)', r'"\1"', content)
    content = re.sub(r"f'([^']*?)'(?![^']*\{)", r"'\1'", content)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    """Fix common issues across the codebase."""
    base_path = Path("/Users/arigazzi/Documents/DeepLearning/LLM for climate/HPE-LLM4Climate")

    # Get list of Python files with f-string warnings
    result = subprocess.run(
        ["pylint", ".", "--disable=all", "--enable=f-string-without-interpolation"],
        cwd=base_path,
        capture_output=True,
        text=True,
        check=False,
    )

    files_to_fix: set[Path] = set()
    for line in result.stdout.split("\n"):
        if "W1309" in line and ".py:" in line:
            parts = line.split(":")
            if parts:
                filepath_str = parts[0].strip()
                if filepath_str.startswith("multimodal_aifs"):
                    files_to_fix.add(base_path / filepath_str)

    print(f"Found {len(files_to_fix)} files with f-string issues")

    fixed = 0
    for filepath in files_to_fix:
        if fix_f_strings_in_file(filepath):
            print(f"  Fixed: {filepath.relative_to(base_path)}")
            fixed += 1

    print(f"\nFixed {fixed} files")


if __name__ == "__main__":
    main()
