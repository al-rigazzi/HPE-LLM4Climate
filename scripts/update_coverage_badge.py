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
"""Utility to synchronize the README coverage badge with coverage.xml."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BADGE_PATTERN = re.compile(
    r"\[!\[Coverage\]\(https://img\.shields\.io/badge/coverage-[^)]*%25-[^)]*\.svg\)\]\(coverage\.xml\)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        default="coverage.xml",
        type=Path,
        help="Path to coverage XML report produced by coverage.py",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        type=Path,
        help="Markdown file containing the coverage badge",
    )
    return parser.parse_args()


def _read_line_rate(xml_path: Path) -> float:
    if not xml_path.exists():
        raise FileNotFoundError(f"Coverage report not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        return float(root.attrib["line-rate"])
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError("coverage.xml missing 'line-rate' attribute") from exc


def _format_percent(line_rate: float) -> str:
    percent = line_rate * 100.0
    return f"{percent:.1f}".rstrip("0").rstrip(".")


def _badge_color(percent_value: float) -> str:
    if percent_value >= 90.0:
        return "brightgreen"
    if percent_value >= 80.0:
        return "green"
    if percent_value >= 70.0:
        return "yellow"
    if percent_value >= 60.0:
        return "orange"
    return "red"


def _build_badge(percent_label: str, percent_value: float) -> str:
    color = _badge_color(percent_value)
    return (
        f"[![Coverage](https://img.shields.io/badge/coverage-"
        f"{percent_label}%25-{color}.svg)](coverage.xml)"
    )


def _update_readme(readme_path: Path, new_badge: str) -> None:
    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")
    content = readme_path.read_text(encoding="utf-8")
    if BADGE_PATTERN.search(content):
        updated = BADGE_PATTERN.sub(new_badge, content, count=1)
    else:
        license_anchor = "[![License"
        idx = content.find(license_anchor)
        if idx == -1:
            raise RuntimeError("Unable to find insertion point for coverage badge")
        insert_pos = content.index("\n", idx) + 1
        updated = content[:insert_pos] + new_badge + "\n" + content[insert_pos:]
    readme_path.write_text(updated, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    line_rate = _read_line_rate(args.coverage)
    percent_label = _format_percent(line_rate)
    badge = _build_badge(percent_label, line_rate * 100.0)
    _update_readme(args.readme, badge)
    print(f"Updated coverage badge to {percent_label}% from {args.coverage}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
