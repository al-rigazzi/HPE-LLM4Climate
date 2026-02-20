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
Climate prompt generator using Jinja2 templates.

This module creates well-formatted prompts combining location information
and climate statistics for querying the Mistral model using template files.
"""

from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader, TemplateNotFound

from .location_masks import Location
from .statistics_computer import VariableStatistics


class ClimatePromptGenerator:
    """
    Generates prompts for climate analysis tasks using Jinja2 templates.

    This class loads and renders prompt templates that combine:
    1. Location information
    2. Climate statistics tables
    3. Task-specific instructions
    """

    def __init__(self, templates_dir: Path | None = None):
        """
        Initialize the prompt generator with Jinja2 templates.

        Args:
            templates_dir: Directory containing Jinja2 template files
                         (defaults to prompts/ subdirectory)
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "prompts"

        self.templates_dir = Path(templates_dir).resolve()
        self._template_sources = self._load_template_sources()
        self.env = Environment(
            loader=ChoiceLoader(
                [
                    DictLoader(self._template_sources),
                    FileSystemLoader(str(self.templates_dir)),
                ]
            ),
            trim_blocks=True,
            lstrip_blocks=True,
            auto_reload=False,
        )

    def _load_template_sources(self) -> dict[str, str]:
        """Load all local Jinja2 templates into memory for robust runtime access."""
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

        template_sources: dict[str, str] = {}
        for template_path in sorted(self.templates_dir.glob("*.jinja2")):
            template_sources[template_path.name] = template_path.read_text(encoding="utf-8")

        if not template_sources:
            raise FileNotFoundError(f"No Jinja2 templates found in directory: {self.templates_dir}")

        return template_sources

    def _reload_templates(self) -> None:
        """Reload templates and refresh loaders after transient filesystem issues."""
        self._template_sources = self._load_template_sources()
        self.env.loader = ChoiceLoader(
            [
                DictLoader(self._template_sources),
                FileSystemLoader(str(self.templates_dir)),
            ]
        )

    def _get_template(self, template_name: str):
        """Get template with a single reload retry for transient shared-filesystem misses."""
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            self._reload_templates()
            try:
                return self.env.get_template(template_name)
            except TemplateNotFound as retry_exc:
                available = sorted(self._template_sources.keys())
                raise FileNotFoundError(
                    f"Template '{template_name}' not found in {self.templates_dir}. "
                    f"Available templates: {available}"
                ) from retry_exc

    def generate_analysis_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
        task_type: str = "weather_description",
    ) -> str:
        """
        Generate a prompt for climate analysis using templates.

        Args:
            location: Location object with mask and metadata
            statistics: List of variable statistics
            statistics_table: Formatted statistics table
            task_type: Type of task
                ("weather_description", "forecast", "anomaly_detection", "basic")

        Returns:
            Formatted prompt string
        """
        # Map task types to template files
        template_map = {
            "weather_description": "weather_description.jinja2",
            "forecast": "forecast.jinja2",
            "anomaly_detection": "anomaly_detection.jinja2",
            "basic": "basic.jinja2",
        }

        if task_type not in template_map:
            raise ValueError(
                f"Unknown task type: {task_type}. Available: {list(template_map.keys())}"
            )

        template_name = template_map[task_type]
        template = self._get_template(template_name)

        # Build context summary for weather description
        context_summary = None
        if task_type == "weather_description":
            context_summary = self._build_context_summary(statistics)

        # Handle basic Q&A task differently
        if task_type == "basic":
            question, answer = self.generate_basic_qa_prompt(location, statistics, statistics_table)
            return template.render(
                question=question,
                answer=answer,
                location=location,
                statistics=statistics,
                statistics_table=statistics_table,
            )

        return template.render(
            location=location,
            statistics=statistics,
            statistics_table=statistics_table,
            context_summary=context_summary,
        )

    def generate_rl_training_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> str:
        """
        Generate a prompt specifically for RL training.

        Args:
            location: Location object
            statistics: Variable statistics (unused but kept for compatibility)
            statistics_table: Formatted statistics table

        Returns:
            Training-optimized prompt
        """
        template = self._get_template("rl_training.jinja2")
        return template.render(
            location=location,
            statistics_table=statistics_table,
        )

    def format_for_mistral(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> str:
        """
        Format prompt in Mistral's chat format using template.

        Args:
            prompt: The main prompt
            system_message: Optional system message

        Returns:
            Formatted prompt string
        """
        template = self._get_template("mistral_chat_wrapper.jinja2")
        return template.render(
            prompt=prompt,
            system_message=system_message,
        )

    def generate_basic_qa_prompt(
        self,
        location: Location,
        statistics: list[VariableStatistics],
        statistics_table: str,
    ) -> tuple[str, str]:
        """
        Generate a basic Q&A pair from climate statistics.

        Args:
            location: Location object with mask and metadata
            statistics: List of variable statistics
            statistics_table: Formatted statistics table

        Returns:
            Tuple of (question, answer) strings
        """
        import random

        if not statistics:
            raise ValueError("No statistics available for generating basic Q&A")

        # Select a random statistic to ask about
        stat = random.choice(statistics)

        # Variable name mappings for human-readable questions
        variable_descriptions = {
            "2t": "temperature at 2m",
            "10u": "10m u-component wind",
            "10v": "10m v-component wind",
            "msl": "mean sea level pressure",
            "tp": "total precipitation",
            "skt": "skin temperature",
            "sp": "surface pressure",
            "tcw": "total column water",
        }

        # Add pressure level variables
        for level in [
            "50",
            "100",
            "150",
            "200",
            "250",
            "300",
            "400",
            "500",
            "600",
            "700",
            "850",
            "925",
            "1000",
        ]:
            for param in ["t", "u", "v", "q", "z", "w"]:
                var_name = f"{param}_{level}"
                param_names = {
                    "t": "temperature",
                    "u": "u-component wind",
                    "v": "v-component wind",
                    "q": "specific humidity",
                    "z": "geopotential height",
                    "w": "vertical velocity",
                }
                variable_descriptions[var_name] = f"{param_names[param]} at {level} hPa"

        # Get description for the selected variable
        var_description = variable_descriptions.get(stat.variable_name, stat.variable_name)

        # Generate question and answer based on statistic type
        statistic_types = ["minimum", "maximum", "average"]
        stat_type = random.choice(statistic_types)

        # Add bounding box coordinates to location description for better alignment
        location_with_coords = location.description
        bbox_str = location.format_bbox_string()
        if bbox_str:
            location_with_coords = f"{location.description} (bbox: {bbox_str})"

        response_format = "Answer with a single numeric value and unit only."

        if stat_type == "minimum":
            value = stat.min_value
            question = (
                f"What is the minimum {var_description} in {location_with_coords}? "
                f"{response_format}"
            )

        elif stat_type == "maximum":
            value = stat.max_value
            question = (
                f"What is the maximum {var_description} in {location_with_coords}? "
                f"{response_format}"
            )

        else:  # average
            value = stat.mean_value
            question = (
                f"What is the average {var_description} in {location_with_coords}? "
                f"{response_format}"
            )

        # Format the value with appropriate units and precision
        formatted_value = self._format_value_with_units(value, stat.variable_name)

        # Generate answer
        answer = (
            f"The {stat_type} {var_description} in {location.description} is {formatted_value}."
        )

        return question, answer

    def _format_value_with_units(self, value: float, variable_name: str) -> str:
        """Format a value with appropriate units and precision."""
        # Temperature variables (convert K to °C)
        if variable_name in ["2t", "skt"] or variable_name.startswith("t_"):
            return f"{value - 273.15:.1f}°C"

        # Wind components (m/s)
        if variable_name in ["10u", "10v"] or variable_name.startswith(("u_", "v_", "w_")):
            return f"{value:.1f} m/s"

        # Height variables (m)
        if variable_name.startswith("z_"):
            return f"{value:.0f} m"

        # Humidity
        if variable_name.startswith("q_"):
            return f"{value:.6f} kg/kg"

        # Special variables with unit conversions
        special_formats = {
            "tp": (lambda v: v * 1000, "{:.2f} mm"),
            "tcw": (lambda v: v, "{:.1f} kg/m²"),
            "msl": (lambda v: v / 100, "{:.1f} hPa"),
            "sp": (lambda v: v / 100, "{:.1f} hPa"),
        }

        if variable_name in special_formats:
            converter, fmt = special_formats[variable_name]
            return fmt.format(converter(value))

        # Default formatting
        return f"{value:.3f}"

    def _build_context_summary(self, statistics: list[VariableStatistics]) -> str:
        """
        Build a context summary from key weather variables.

        Args:
            statistics: List of variable statistics

        Returns:
            Human-readable summary string
        """
        # Get key weather variables
        temp_stats = [s for s in statistics if "2t" in s.variable_name]
        wind_stats = [s for s in statistics if "10u" in s.variable_name or "10v" in s.variable_name]
        precip_stats = [s for s in statistics if "tp" in s.variable_name]
        pressure_stats = [s for s in statistics if "msl" in s.variable_name]

        # Build context parts
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

        return ", ".join(context_parts) if context_parts else "various weather conditions"

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
