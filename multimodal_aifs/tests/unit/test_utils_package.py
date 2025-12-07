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
"""Tests for ``multimodal_aifs.utils`` package helpers."""

from __future__ import annotations

import multimodal_aifs.utils as utils_pkg


def test_get_utils_info_reports_components():
    """Package metadata should list all utility components."""

    info = utils_pkg.get_utils_info()
    assert info["package"] == "multimodal_aifs.utils"
    assert set(info["components"].keys()) >= {"climate_data", "location", "text"}


def test_test_all_utils_runs_with_monkeypatch(monkeypatch):
    """``test_all_utils`` should orchestrate sub-tests when they are available."""

    called = []

    def fake_climate():
        called.append("climate")

    def fake_location():
        called.append("location")

    def fake_text():
        called.append("text")

    monkeypatch.setattr(
        "multimodal_aifs.utils.climate_data_utils.test_climate_processor",
        fake_climate,
        raising=False,
    )
    monkeypatch.setattr(
        "multimodal_aifs.utils.location_utils.test_location_utilities",
        fake_location,
        raising=False,
    )
    monkeypatch.setattr(
        "multimodal_aifs.utils.text_utils.test_text_processing",
        fake_text,
        raising=False,
    )

    utils_pkg.test_all_utils()
    assert called == ["climate", "location", "text"]
