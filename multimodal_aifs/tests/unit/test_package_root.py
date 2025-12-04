#!/usr/bin/env python3
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
"""Tests for package-level helpers in ``multimodal_aifs``."""

from __future__ import annotations

import builtins

import multimodal_aifs as mma


def test_get_package_info_contents():
    """Package info should expose metadata used by downstream tools."""

    info = mma.get_package_info()
    assert info["name"] == "multimodal_aifs"
    assert info["version"] == mma.__version__
    assert info["aifs_integration"] is True


def test_check_requirements_handles_missing_dependency(monkeypatch):
    """``check_requirements`` should return False when optional deps are missing."""

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("anemoi"):
            raise ImportError("missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(mma.check_requirements.__globals__, "__import__", fake_import)
    assert mma.check_requirements() is False


def test_check_requirements_success_when_all_present(monkeypatch):
    """When imports succeed, ``check_requirements`` should return True."""

    class DummyModule:  # pylint: disable=too-few-public-methods
        """Lightweight stand-in that exposes nested attributes."""

        def __getattr__(self, _):
            return self

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        return DummyModule()

    monkeypatch.setitem(mma.check_requirements.__globals__, "__import__", fake_import)
    assert mma.check_requirements() is True
