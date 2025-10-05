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
Climate prompt generator for Mistral language model.

This module creates well-formatted prompts combining location information
and climate statistics for querying the Mistral model.
"""

from .location_masks import Location
from .statistics_computer import VariableStatistics


class ClimatePromptGenerator:
    """
    Generates prompts for climate analysis tasks.

    This class creates natural language prompts that combine:
    1. Location information
    2. Climate statistics tables
    3. Task instructions
    """

    def __init__(self, prompt_style: str = "detailed"):
        """
        Initialize the prompt generator.

        Args:
            prompt_style: Style of prompts ("detailed", "concise", or "technical")
        """
        self.prompt_style = prompt_style

    def generate_analysis_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
        task_type: str = "weather_description",
    ) -> str:
        """
        Generate a prompt for climate analysis.

        Args:
            location: Location object with mask and metadata
            statistics: List of variable statistics
            statistics_table: Formatted statistics table
            task_type: Type of task ("weather_description", "forecast", "anomaly_detection")

        Returns:
            Formatted prompt string
        """
        if task_type == "weather_description":
            return self._generate_weather_description_prompt(location, statistics, statistics_table)
        elif task_type == "forecast":
            return self._generate_forecast_prompt(location, statistics, statistics_table)
        elif task_type == "anomaly_detection":
            return self._generate_anomaly_prompt(location, statistics, statistics_table)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _generate_weather_description_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> str:
        """Generate a weather description prompt."""

        # Get key weather variables
        temp_stats = [s for s in statistics if "2t" in s.variable_name]
        wind_stats = [s for s in statistics if "10u" in s.variable_name or "10v" in s.variable_name]
        precip_stats = [s for s in statistics if "tp" in s.variable_name]
        pressure_stats = [s for s in statistics if "msl" in s.variable_name]

        # Build context
        context_parts = []

        if temp_stats:
            temp = temp_stats[0].mean_value - 273.15  # Convert K to °C
            context_parts.append(f"average temperature of {temp:.1f}°C")

        if wind_stats and len(wind_stats) >= 2:
            u_wind = wind_stats[0].mean_value
            v_wind = wind_stats[1].mean_value if len(wind_stats) > 1 else 0
            wind_speed = (u_wind**2 + v_wind**2) ** 0.5
            context_parts.append(f"wind speed of {wind_speed:.1f} m/s")

        if precip_stats:
            precip = precip_stats[0].mean_value * 1000  # Convert m to mm
            context_parts.append(f"precipitation of {precip:.1f} mm")

        if pressure_stats:
            pressure = pressure_stats[0].mean_value / 100  # Convert Pa to hPa
            context_parts.append(f"sea level pressure of {pressure:.1f} hPa")

        context_summary = (
            ", ".join(context_parts) if context_parts else "various weather conditions"
        )

        prompt = f"""You are an expert meteorologist analyzing climate data for {location.description}.

Location Details:
- Name: {location.name}
- Type: {location.location_type}
- Coordinates: {location.center_lat:.2f}°N, {location.center_lon:.2f}°E
- Current conditions: {context_summary}

Below is a detailed statistical analysis of the atmospheric variables in this region:

{statistics_table}

Task: Based on the provided climate data, describe the current weather conditions in {location.description}. Include:
1. Temperature conditions and thermal comfort
2. Wind patterns and air circulation
3. Precipitation and moisture levels
4. Pressure systems and their implications
5. Any notable atmospheric features or patterns

Provide a comprehensive yet accessible description suitable for both meteorologists and general public."""

        return prompt

    def _generate_forecast_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> str:
        """Generate a forecast analysis prompt."""

        prompt = f"""You are an expert meteorological forecaster analyzing current conditions for {location.description}.

Location Details:
- Name: {location.name}
- Type: {location.location_type}
- Coordinates: {location.center_lat:.2f}°N, {location.center_lon:.2f}°E

Current atmospheric state analysis:

{statistics_table}

Task: Based on these current atmospheric conditions, provide a short-term forecast analysis for {location.description}. Consider:
1. Trends in temperature and pressure systems
2. Wind patterns and their evolution
3. Moisture content and precipitation potential
4. Atmospheric stability and convection potential
5. Expected weather developments in the next 6-12 hours

Focus on physical meteorological reasoning based on the observed variables."""

        return prompt

    def _generate_anomaly_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> str:
        """Generate an anomaly detection prompt."""

        prompt = f"""You are a climate scientist investigating atmospheric anomalies in {location.description}.

Location Details:
- Name: {location.name}
- Type: {location.location_type}
- Coordinates: {location.center_lat:.2f}°N, {location.center_lon:.2f}°E

Atmospheric analysis data:

{statistics_table}

Task: Analyze the atmospheric data for {location.description} and identify any unusual patterns or anomalies. Consider:
1. Temperature extremes or unusual gradients
2. Abnormal pressure patterns
3. Unusual wind circulation or divergence/convergence
4. Moisture anomalies
5. Any indicators of severe weather potential

Explain the physical significance of any anomalies detected and their potential impacts."""

        return prompt

    def generate_multimodal_training_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> str:
        """
        Generate a prompt specifically for multimodal training.

        This creates prompts that encourage the model to integrate
        spatial climate data with textual descriptions.

        Args:
            location: Location object
            statistics: Variable statistics
            statistics_table: Formatted table

        Returns:
            Training-optimized prompt
        """
        prompt = f"""Analyze the weather conditions in {location.description} based on the following climate data:

{statistics_table}

Describe the weather conditions, including temperature, wind, precipitation, and any notable atmospheric patterns."""

        return prompt

    def create_instruction_format(
        self,
        location: Location,
        statistics_table: str,
        instruction: str,
    ) -> tuple[str, str]:
        """
        Create instruction-following format for training.

        Args:
            location: Location object
            statistics_table: Formatted table
            instruction: Custom instruction

        Returns:
            Tuple of (instruction, input_data)
        """
        input_data = f"""Location: {location.description}
Coordinates: {location.center_lat:.2f}°N, {location.center_lon:.2f}°E

Climate Data:
{statistics_table}"""

        return instruction, input_data

    def format_for_mistral(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> str:
        """
        Format prompt in Mistral's chat format.

        Args:
            prompt: The main prompt
            system_message: Optional system message

        Returns:
            Formatted prompt string
        """
        if system_message:
            formatted = f"""<s>[INST] {system_message}

{prompt} [/INST]"""
        else:
            formatted = f"<s>[INST] {prompt} [/INST]"

        return formatted

    def extract_key_insights(self, statistics: list[VariableStatistics]) -> dict[str, str]:
        """
        Extract key meteorological insights from statistics.

        Args:
            statistics: List of variable statistics

        Returns:
            Dictionary of insight categories to descriptions
        """
        insights = {}

        # Temperature analysis
        temp_stats = [s for s in statistics if "2t" == s.variable_name]
        if temp_stats:
            temp_c = temp_stats[0].mean_value - 273.15
            if temp_c < 0:
                insights["temperature"] = f"Freezing conditions ({temp_c:.1f}°C)"
            elif temp_c < 10:
                insights["temperature"] = f"Cold ({temp_c:.1f}°C)"
            elif temp_c < 25:
                insights["temperature"] = f"Mild ({temp_c:.1f}°C)"
            else:
                insights["temperature"] = f"Warm ({temp_c:.1f}°C)"

        # Wind analysis
        wind_u = [s for s in statistics if "10u" == s.variable_name]
        wind_v = [s for s in statistics if "10v" == s.variable_name]
        if wind_u and wind_v:
            speed = (wind_u[0].mean_value ** 2 + wind_v[0].mean_value ** 2) ** 0.5
            if speed < 3:
                insights["wind"] = f"Light winds ({speed:.1f} m/s)"
            elif speed < 8:
                insights["wind"] = f"Moderate winds ({speed:.1f} m/s)"
            else:
                insights["wind"] = f"Strong winds ({speed:.1f} m/s)"

        # Pressure analysis
        pressure_stats = [s for s in statistics if "msl" == s.variable_name]
        if pressure_stats:
            pressure_hpa = pressure_stats[0].mean_value / 100
            if pressure_hpa < 1000:
                insights["pressure"] = f"Low pressure system ({pressure_hpa:.0f} hPa)"
            elif pressure_hpa > 1020:
                insights["pressure"] = f"High pressure system ({pressure_hpa:.0f} hPa)"
            else:
                insights["pressure"] = f"Normal pressure ({pressure_hpa:.0f} hPa)"

        return insights
