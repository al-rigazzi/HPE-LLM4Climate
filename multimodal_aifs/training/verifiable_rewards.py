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
"""
Verifiable Reward Computation for Climate LLM Training.

This module provides reward functions that compare LLM-generated text responses
against ground-truth numerical statistics computed from climate data. The rewards
are verifiable because they are derived from numerical computations that can be
independently validated.

Key reward components:
1. Numerical accuracy - comparing extracted values to ground truth
2. Trend correctness - verifying qualitative descriptions match data
3. Coverage - ensuring all relevant variables are mentioned
4. Consistency - internal consistency of the response
"""

import re
from dataclasses import dataclass, field

import torch

from .statistics_computer import VariableStatistics


@dataclass
class ExtractedValue:
    """A numerical value extracted from text with context."""

    value: float
    unit: str
    variable_hint: str
    context: str
    position: int


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    numerical_accuracy: float = 0.0
    trend_correctness: float = 0.0
    coverage: float = 0.0
    consistency: float = 0.0
    format_quality: float = 0.0

    total: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute total reward after initialization."""
        self.total = (
            0.30 * self.numerical_accuracy
            + 0.20 * self.trend_correctness
            + 0.30 * self.coverage
            + 0.05 * self.consistency
            + 0.15 * self.format_quality
        )


class VerifiableRewardComputer:
    """
    Computes verifiable rewards by comparing LLM outputs to ground truth statistics.

    The reward is "verifiable" because it is based on numerical comparisons that
    can be independently validated by re-computing the statistics from the raw data.

    Reward components:
    - Numerical accuracy (40%): How close are mentioned numbers to ground truth
    - Trend correctness (25%): Do qualitative descriptions match data trends
    - Coverage (20%): Are all important variables addressed
    - Consistency (10%): Is the response internally consistent
    - Format quality (5%): Proper units, structure
    """

    UNIT_MAPPINGS = {
        "temperature": ["k", "kelvin", "celsius", "c", "°c", "°k", "fahrenheit", "f", "°f"],
        "pressure": ["pa", "pascal", "hpa", "mbar", "millibar", "mb"],
        "wind": ["m/s", "ms-1", "meters per second", "knots", "kts", "mph", "km/h", "kph"],
        "moisture": ["kg/kg", "kg/m²", "kg/m2", "mm", "percent", "%"],
        "geopotential": ["m²/s²", "m2/s2", "j/kg", "gpm", "dam"],
    }

    TREND_KEYWORDS = {
        "increasing": ["increasing", "rising", "growing", "higher", "elevated", "above"],
        "decreasing": ["decreasing", "falling", "dropping", "lower", "reduced", "below"],
        "stable": ["stable", "steady", "constant", "unchanged", "normal"],
        "variable": ["variable", "fluctuating", "varying", "unstable", "changing"],
    }

    IMPORTANT_VARIABLES = [
        "2t",
        "msl",
        "10u",
        "10v",
        "tcw",
        "sp",
        "z_500",
        "t_850",
        "q_700",
    ]

    def __init__(
        self,
        numerical_tolerance: float = 0.1,
        use_relative_error: bool = True,
        penalize_hallucinations: bool = True,
        reward_scale: float = 1.0,
        reward_mode: str = "full",
    ):
        """
        Initialize the reward computer.

        Args:
            numerical_tolerance: Tolerance for numerical comparisons
                                 (relative if use_relative_error)
            use_relative_error: If True, use relative error; otherwise absolute
            penalize_hallucinations: If True, penalize values not in ground truth
            reward_scale: Scale factor for final reward
            reward_mode: "full" (all components) or "rl_only" (numerical accuracy only)
        """
        self.numerical_tolerance = numerical_tolerance
        self.use_relative_error = use_relative_error
        self.penalize_hallucinations = penalize_hallucinations
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode

        self._variable_patterns = self._compile_variable_patterns()

    def _compile_variable_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for variable name detection."""
        patterns = {}

        name_mappings = {
            "2t": r"(?:2[- ]?m(?:eter)?|surface|near[- ]surface)\s*temp(?:erature)?",
            "msl": r"(?:mean\s+)?sea[- ]level\s+pressure|msl|mslp",
            "10u": r"(?:10[- ]?m(?:eter)?)\s*(?:u[- ]?)?wind|zonal\s+wind|u[- ]component",
            "10v": r"(?:10[- ]?m(?:eter)?)\s*(?:v[- ]?)?wind|meridional\s+wind|v[- ]component",
            "tcw": r"total\s+column\s+water|tcw|precipitable\s+water",
            "sp": r"surface\s+pressure",
            "skt": r"skin\s+temp(?:erature)?",
        }

        for level in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]:
            name_mappings[f"z_{level}"] = (
                rf"(?:geopotential|height).*{level}\s*(?:hpa|mb)|"
                rf"{level}\s*(?:hpa|mb).*(?:geopotential|height)"
            )
            name_mappings[f"t_{level}"] = (
                rf"temp(?:erature)?.*{level}\s*(?:hpa|mb)|"
                rf"{level}\s*(?:hpa|mb).*temp(?:erature)?"
            )
            name_mappings[f"q_{level}"] = (
                rf"(?:specific\s+)?humidity.*{level}\s*(?:hpa|mb)|"
                rf"{level}\s*(?:hpa|mb).*humidity"
            )
            name_mappings[f"u_{level}"] = (
                rf"(?:u[- ]?wind|zonal).*{level}\s*(?:hpa|mb)|"
                rf"{level}\s*(?:hpa|mb).*(?:u[- ]?wind|zonal)"
            )
            name_mappings[f"v_{level}"] = (
                rf"(?:v[- ]?wind|meridional).*{level}\s*(?:hpa|mb)|"
                rf"{level}\s*(?:hpa|mb).*(?:v[- ]?wind|meridional)"
            )

        for var_name, pattern in name_mappings.items():
            patterns[var_name] = re.compile(pattern, re.IGNORECASE)

        return patterns

    def extract_numerical_values(self, text: str) -> list[ExtractedValue]:
        """
        Extract numerical values with units and context from text.

        Args:
            text: LLM response text

        Returns:
            List of extracted values with metadata
        """
        values = []

        number_pattern = r"(-?\d+\.?\d*)\s*([A-Za-z°%/²]+(?:/[A-Za-z²]+)?)"
        matches = list(re.finditer(number_pattern, text))

        for match in matches:
            try:
                value = float(match.group(1))
                unit = match.group(2).lower()

                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                variable_hint = self._identify_variable_from_context(context, unit)

                values.append(
                    ExtractedValue(
                        value=value,
                        unit=unit,
                        variable_hint=variable_hint,
                        context=context,
                        position=match.start(),
                    )
                )
            except ValueError:
                continue

        return values

    def _identify_variable_from_context(self, context: str, unit: str) -> str:
        """Identify which climate variable a value refers to based on context."""
        for var_name, pattern in self._variable_patterns.items():
            if pattern.search(context):
                return var_name

        # Mapping from variable type to default variable name
        type_to_var = {
            "temperature": "2t",
            "pressure": "msl",
            "wind": "10u",
            "moisture": "tcw",
            "geopotential": "z_500",
        }

        for var_type, units in self.UNIT_MAPPINGS.items():
            if unit in units and var_type in type_to_var:
                return type_to_var[var_type]

        return "unknown"

    def compute_numerical_accuracy(
        self,
        extracted_values: list[ExtractedValue],
        ground_truth: list[VariableStatistics],
    ) -> tuple[float, dict]:
        """
        Compute numerical accuracy reward.

        Args:
            extracted_values: Values extracted from LLM response
            ground_truth: Ground truth statistics

        Returns:
            Tuple of (reward, details_dict)
        """
        if not extracted_values:
            return 0.0, {"reason": "no_values_extracted"}

        gt_lookup = {stat.variable_name: stat for stat in ground_truth}

        correct_count = 0
        total_count = 0
        hallucination_count = 0
        details: dict = {"matches": [], "misses": [], "hallucinations": []}

        for extracted in extracted_values:
            var_name = extracted.variable_hint
            if var_name == "unknown":
                continue

            total_count += 1

            if var_name in gt_lookup:
                gt_stat = gt_lookup[var_name]

                is_correct, error = self._check_value_accuracy(
                    extracted.value,
                    gt_stat.min_value,
                    gt_stat.max_value,
                    gt_stat.mean_value,
                )

                if is_correct:
                    correct_count += 1
                    details["matches"].append(
                        {
                            "variable": var_name,
                            "extracted": extracted.value,
                            "error": error,
                        }
                    )
                else:
                    details["misses"].append(
                        {
                            "variable": var_name,
                            "extracted": extracted.value,
                            "expected_range": [gt_stat.min_value, gt_stat.max_value],
                            "error": error,
                        }
                    )
            elif self.penalize_hallucinations:
                hallucination_count += 1
                details["hallucinations"].append(
                    {
                        "variable": var_name,
                        "value": extracted.value,
                    }
                )

        if total_count == 0:
            return 0.0, {"reason": "no_identifiable_values"}

        accuracy = correct_count / total_count
        if self.penalize_hallucinations and hallucination_count > 0:
            penalty = 0.1 * hallucination_count / total_count
            accuracy = max(0.0, accuracy - penalty)

        return accuracy, details

    def _check_value_accuracy(
        self,
        extracted: float,
        gt_min: float,
        gt_max: float,
        gt_mean: float,
    ) -> tuple[bool, float]:
        """
        Check if an extracted value is accurate.

        Returns True if the value falls within an acceptable range of the
        ground truth statistics.
        """
        if gt_min <= extracted <= gt_max:
            return True, 0.0

        if self.use_relative_error:
            if abs(gt_mean) < 1e-10:
                reference = max(abs(gt_max - gt_min), 1.0)
            else:
                reference = abs(gt_mean)
            error = min(abs(extracted - gt_min), abs(extracted - gt_max)) / reference
        else:
            error = min(abs(extracted - gt_min), abs(extracted - gt_max))

        return error <= self.numerical_tolerance, error

    def compute_trend_correctness(
        self,
        text: str,
        ground_truth: list[VariableStatistics],
    ) -> tuple[float, dict]:
        """
        Compute trend correctness reward.

        Checks if qualitative trend descriptions match the actual data.

        Args:
            text: LLM response text
            ground_truth: Ground truth statistics

        Returns:
            Tuple of (reward, details_dict)
        """
        text_lower = text.lower()
        correct_trends = 0
        total_trends = 0
        details: dict = {"correct": [], "incorrect": []}

        for stat in ground_truth:
            if stat.std_value < 1e-10:
                actual_trend = "stable"
            elif stat.mean_value > (stat.min_value + stat.max_value) / 2 * 1.1:
                actual_trend = "increasing"
            elif stat.mean_value < (stat.min_value + stat.max_value) / 2 * 0.9:
                actual_trend = "decreasing"
            else:
                actual_trend = "variable"

            pattern = self._variable_patterns.get(stat.variable_name)
            if pattern and pattern.search(text):
                total_trends += 1

                mentioned_trends = []
                for trend_type, keywords in self.TREND_KEYWORDS.items():
                    if any(kw in text_lower for kw in keywords):
                        mentioned_trends.append(trend_type)

                if not mentioned_trends or actual_trend in mentioned_trends:
                    correct_trends += 1
                    details["correct"].append(
                        {
                            "variable": stat.variable_name,
                            "actual": actual_trend,
                            "mentioned": mentioned_trends,
                        }
                    )
                else:
                    details["incorrect"].append(
                        {
                            "variable": stat.variable_name,
                            "actual": actual_trend,
                            "mentioned": mentioned_trends,
                        }
                    )

        if total_trends == 0:
            return 0.5, {"reason": "no_trend_mentions"}

        return correct_trends / total_trends, details

    def compute_coverage(
        self,
        text: str,
        ground_truth: list[VariableStatistics],
    ) -> tuple[float, dict]:
        """
        Compute coverage reward.

        Checks how many important variables are mentioned in the response.

        Args:
            text: LLM response text
            ground_truth: Ground truth statistics

        Returns:
            Tuple of (reward, details_dict)
        """
        available_important = [
            v
            for v in self.IMPORTANT_VARIABLES
            if any(stat.variable_name == v for stat in ground_truth)
        ]

        if not available_important:
            return 1.0, {"reason": "no_important_variables_in_data"}

        mentioned_count = 0
        mentioned_vars = []
        missing_vars = []

        for var_name in available_important:
            pattern = self._variable_patterns.get(var_name)
            if pattern and pattern.search(text):
                mentioned_count += 1
                mentioned_vars.append(var_name)
            else:
                missing_vars.append(var_name)

        coverage = mentioned_count / len(available_important)

        return coverage, {
            "mentioned": mentioned_vars,
            "missing": missing_vars,
            "coverage_ratio": coverage,
        }

    def compute_consistency(
        self,
        text: str,
        extracted_values: list[ExtractedValue],
    ) -> tuple[float, dict]:
        """
        Compute internal consistency reward.

        Checks if the response is internally consistent (no contradictions).

        Args:
            text: LLM response text
            extracted_values: Previously extracted values

        Returns:
            Tuple of (reward, details_dict)
        """
        values_by_variable: dict[str, list[float]] = {}
        for extracted in extracted_values:
            if extracted.variable_hint != "unknown":
                if extracted.variable_hint not in values_by_variable:
                    values_by_variable[extracted.variable_hint] = []
                values_by_variable[extracted.variable_hint].append(extracted.value)

        inconsistencies = []
        for var_name, values in values_by_variable.items():
            if len(values) > 1:
                max_val = max(values)
                min_val = min(values)
                if abs(max_val) > 1e-10:
                    spread = (max_val - min_val) / abs(max_val)
                else:
                    spread = max_val - min_val

                if spread > 0.5:
                    inconsistencies.append(
                        {
                            "variable": var_name,
                            "values": values,
                            "spread": spread,
                        }
                    )

        if not values_by_variable:
            return 0.5, {"reason": "no_values_to_check"}

        if inconsistencies:
            penalty = 0.2 * len(inconsistencies)
            return max(0.0, 1.0 - penalty), {"inconsistencies": inconsistencies}

        return 1.0, {"consistent": True}

    def compute_format_quality(self, text: str) -> tuple[float, dict]:
        """
        Compute format quality reward.

        Checks for proper formatting and penalises responses that
        echo structured input formats (tables, JSON, key-value lines)
        instead of producing natural-language narrative.

        Args:
            text: LLM response text

        Returns:
            Tuple of (reward, details_dict)
        """
        score = 0.0
        details: dict = {"checks": {}}

        has_units = bool(re.search(r"\d+\.?\d*\s*[A-Za-z°%]+", text))
        details["checks"]["has_units"] = has_units
        if has_units:
            score += 0.2

        has_structure = bool(re.search(r"[\.\n]", text)) and len(text) > 100
        details["checks"]["has_structure"] = has_structure
        if has_structure:
            score += 0.1

        proper_length = 50 < len(text) < 2000
        details["checks"]["proper_length"] = proper_length
        if proper_length:
            score += 0.1

        no_repetition = len(set(text.split())) / max(len(text.split()), 1) > 0.3
        details["checks"]["no_repetition"] = no_repetition
        if no_repetition:
            score += 0.2

        # Penalise echoing structured input formats
        table_echo = bool(re.search(r"={4,}", text)) or bool(
            re.search(r"Variable\s+Min\s+Max", text, re.IGNORECASE)
        )
        json_echo = text.count("{") > 2 and bool(re.search(r'"\w+"\s*:\s*\{', text))
        kv_echo = len(re.findall(r"\w+:\s*min=.*max=.*mean=", text)) > 1

        details["checks"]["table_echo"] = table_echo
        details["checks"]["json_echo"] = json_echo
        details["checks"]["kv_echo"] = kv_echo

        is_echo = table_echo or json_echo or kv_echo
        details["checks"]["is_echo"] = is_echo

        if not is_echo:
            score += 0.4

        return score, details

    def compute_reward(
        self,
        response: str,
        ground_truth: list[VariableStatistics],
        return_breakdown: bool = False,
    ) -> float | tuple[float, RewardBreakdown, dict]:
        """
        Compute the total verifiable reward.

        Args:
            response: LLM response text
            ground_truth: Ground truth statistics from climate data
            return_breakdown: If True, return detailed breakdown

        Returns:
            Total reward (scaled), optionally with breakdown and details
        """
        extracted_values = self.extract_numerical_values(response)

        num_accuracy, num_details = self.compute_numerical_accuracy(extracted_values, ground_truth)
        trend_correct, trend_details = self.compute_trend_correctness(response, ground_truth)
        coverage, coverage_details = self.compute_coverage(response, ground_truth)
        consistency, consistency_details = self.compute_consistency(response, extracted_values)
        format_quality, format_details = self.compute_format_quality(response)

        # Compute weighted total based on reward_mode
        if self.reward_mode == "rl_only":
            # Stage 1: Focus on numerical accuracy only
            total_weighted = num_accuracy
        elif self.reward_mode == "full":
            # Stage 2+: Balanced reward with all components
            # format_quality penalises echoing (tables/JSON/KV) in responses
            total_weighted = (
                0.30 * num_accuracy
                + 0.20 * trend_correct
                + 0.30 * coverage
                + 0.05 * consistency
                + 0.15 * format_quality
            )
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        breakdown = RewardBreakdown(
            numerical_accuracy=num_accuracy,
            trend_correctness=trend_correct,
            coverage=coverage,
            consistency=consistency,
            format_quality=format_quality,
        )
        # Store computed total for reference
        breakdown.total = total_weighted

        total_reward = total_weighted * self.reward_scale

        # Apply zero-coverage penalty only in full mode
        if self.reward_mode == "full":
            if coverage < 0.01:  # coverage=0 (no important variables mentioned)
                total_reward *= 0.1
            # Apply coverage bonus for high coverage: reward well-structured responses
            elif coverage >= 0.67:  # 6+ out of 9 variables
                total_reward *= 1.1

        if return_breakdown:
            details = {
                "numerical": num_details,
                "trends": trend_details,
                "coverage": coverage_details,
                "consistency": consistency_details,
                "format": format_details,
            }
            return total_reward, breakdown, details

        return total_reward

    def compute_batch_rewards(
        self,
        responses: list[str],
        ground_truths: list[list[VariableStatistics]],
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of responses.

        Args:
            responses: List of LLM response texts
            ground_truths: List of ground truth statistics (one per response)

        Returns:
            Tensor of rewards [batch_size]
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Number of responses ({len(responses)}) must match "
                f"number of ground truths ({len(ground_truths)})"
            )

        rewards = []
        for response, gt in zip(responses, ground_truths):
            reward = self.compute_reward(response, gt, return_breakdown=False)
            assert isinstance(reward, float)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)
