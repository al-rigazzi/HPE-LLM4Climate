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
"""Unit tests for ClimatePromptGenerator with Jinja2 templates."""

# pylint: disable=protected-access
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from multimodal_aifs.training.location_masks import Location
from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator
from multimodal_aifs.training.statistics_computer import VariableStatistics


# pylint: disable=too-many-public-methods
class TestClimatePromptGenerator:
    """Test suite for the Jinja2-based ClimatePromptGenerator."""

    @pytest.fixture
    def mock_location(self):
        """Create a mock location for testing."""
        location = Mock(spec=Location)
        location.name = "Test Location"
        location.location_type = "test_region"
        location.description = "Test Location"
        location.center_lat = 45.0
        location.center_lon = -120.0
        return location

    @pytest.fixture
    def mock_statistics(self):
        """Create mock statistics for testing."""
        stats = [
            VariableStatistics(
                variable_name="2t",
                min_value=285.0,
                max_value=295.0,
                mean_value=290.0,
                std_value=1.0,
                unit="K",
                description="2m temperature",
            ),
            VariableStatistics(
                variable_name="10u",
                min_value=-5.0,
                max_value=10.0,
                mean_value=2.5,
                std_value=1.0,
                unit="m/s",
                description="10m u-component of wind",
            ),
            VariableStatistics(
                variable_name="10v",
                min_value=-3.0,
                max_value=8.0,
                mean_value=1.5,
                std_value=1.0,
                unit="m/s",
                description="10m v-component of wind",
            ),
            VariableStatistics(
                variable_name="msl",
                min_value=99000.0,
                max_value=102000.0,
                mean_value=101000.0,
                std_value=1.0,
                unit="Pa",
                description="mean sea level pressure",
            ),
            VariableStatistics(
                variable_name="tp",
                min_value=0.0,
                max_value=0.005,
                mean_value=0.001,
                std_value=1.0,
                unit="m",
                description="total precipitation",
            ),
        ]
        return stats

    @pytest.fixture
    def statistics_table(self):
        """Create a mock statistics table."""
        return """Variable    Min      Max      Mean     Gradient
2t          285.0    295.0    290.0    0.1
10u         -5.0     10.0     2.5      0.05
10v         -3.0     8.0      1.5      0.04
msl         99000    102000   101000   50.0
tp          0.0      0.005    0.001    0.0001"""

    @pytest.fixture
    def prompt_generator(self):
        """Create a ClimatePromptGenerator instance."""
        return ClimatePromptGenerator()

    def test_initialization_default_templates(self):
        """Test that generator initializes with default template directory."""
        generator = ClimatePromptGenerator()
        # Check that templates directory exists
        assert generator.templates_dir.exists()
        assert generator.templates_dir.name == "prompts"
        # Check that required template files exist
        required_templates = [
            "weather_description.jinja2",
            "forecast.jinja2",
            "anomaly_detection.jinja2",
            "rl_training.jinja2",
            "basic.jinja2",
            "mistral_chat_wrapper.jinja2",
        ]
        for template in required_templates:
            template_path = generator.templates_dir / template
            assert template_path.exists(), f"Template {template} not found"

    def test_initialization_custom_templates_dir(self):
        """Test initialization with custom templates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a test template
            test_template = temp_path / "test.jinja2"
            test_template.write_text("Hello {{ name }}!")
            generator = ClimatePromptGenerator(templates_dir=temp_path)
            assert generator.templates_dir == temp_path

    def test_weather_description_prompt_generation(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test weather description prompt generation."""
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=mock_statistics,
            statistics_table=statistics_table,
            task_type="weather_description",
        )
        # Check prompt structure
        assert "expert meteorologist" in prompt
        assert mock_location.name in prompt
        assert mock_location.location_type in prompt
        assert "45.00°N" in prompt
        assert "-120.00°E" in prompt
        assert statistics_table in prompt
        # Check task-specific content
        assert "Temperature conditions and thermal comfort" in prompt
        assert "Wind patterns and air circulation" in prompt
        assert "Precipitation and moisture levels" in prompt
        assert "Pressure systems" in prompt
        # Check context summary is included
        assert "average temperature" in prompt or "Current conditions" in prompt

    def test_forecast_prompt_generation(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test forecast prompt generation."""
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=mock_statistics,
            statistics_table=statistics_table,
            task_type="forecast",
        )
        # Check prompt structure
        assert "meteorological forecaster" in prompt
        assert mock_location.name in prompt
        assert statistics_table in prompt
        # Check forecast-specific content
        assert "short-term forecast" in prompt
        assert "6-12 hours" in prompt
        assert "Trends in temperature" in prompt
        assert "Wind patterns and their evolution" in prompt

    def test_anomaly_detection_prompt_generation(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test anomaly detection prompt generation."""
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=mock_statistics,
            statistics_table=statistics_table,
            task_type="anomaly_detection",
        )
        # Check prompt structure
        assert "climate scientist" in prompt
        assert "anomalies" in prompt
        assert mock_location.name in prompt
        assert statistics_table in prompt
        # Check anomaly-specific content
        assert "Temperature extremes" in prompt
        assert "Abnormal pressure patterns" in prompt
        assert "severe weather potential" in prompt

    def test_rl_training_prompt_generation(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test RL training prompt generation."""
        prompt = prompt_generator.generate_rl_training_prompt(
            location=mock_location, statistics=mock_statistics, statistics_table=statistics_table
        )
        # Check prompt structure
        assert mock_location.description in prompt
        assert statistics_table in prompt
        # Should request natural language output
        assert "natural language" in prompt
        # Should discourage echoing
        assert "Do NOT reproduce" in prompt
        # Should be concise
        assert len(prompt) < 1500

    def test_basic_prompt_generation(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test basic Q&A prompt generation."""
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=mock_statistics,
            statistics_table=statistics_table,
            task_type="basic",
        )
        # Check that it's a Q&A format
        assert "Q:" in prompt
        assert "A:" in prompt
        # Should contain human-readable variable descriptions
        descriptions = ["temperature at 2m", "wind", "pressure", "precipitation"]
        assert any(desc in prompt for desc in descriptions)
        # Should contain question patterns
        question_patterns = ["What is the", "What are the"]
        assert any(pattern in prompt for pattern in question_patterns)
        # Should contain expected statistical terms
        stat_terms = ["minimum", "maximum", "average"]
        assert any(term in prompt for term in stat_terms)

    def test_basic_prompt_generates_different_questions(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test that basic prompts generate different questions on multiple calls."""
        prompts = []
        for _ in range(10):
            prompt = prompt_generator.generate_analysis_prompt(
                location=mock_location,
                statistics=mock_statistics,
                statistics_table=statistics_table,
                task_type="basic",
            )
            prompts.append(prompt)
        # Should have some variation in questions (not all identical)
        unique_prompts = set(prompts)
        assert len(unique_prompts) > 1, "Basic prompts should show some variation"

    def test_basic_prompt_unit_conversion(self, prompt_generator, mock_location, statistics_table):
        """Test that basic prompts properly convert units."""
        # Create statistics with units that need conversion
        test_stats = [
            VariableStatistics(
                variable_name="2t",
                min_value=273.15,  # 0°C in Kelvin
                max_value=293.15,  # 20°C in Kelvin
                mean_value=283.15,  # 10°C in Kelvin
                std_value=1.0,
                unit="K",
                description="2m temperature",
            ),
            VariableStatistics(
                variable_name="msl",
                min_value=101325.0,  # Standard pressure in Pa
                max_value=101325.0,
                mean_value=101325.0,
                std_value=1.0,
                unit="Pa",
                description="mean sea level pressure",
            ),
            VariableStatistics(
                variable_name="tp",
                min_value=0.001,  # 1mm in meters
                max_value=0.001,
                mean_value=0.001,
                std_value=1.0,
                unit="m",
                description="total precipitation",
            ),
        ]
        # Generate multiple prompts to test conversions
        found_temp = False
        found_pressure = False
        found_precip = False
        for _ in range(50):
            prompt = prompt_generator.generate_analysis_prompt(
                location=mock_location,
                statistics=test_stats,
                statistics_table=statistics_table,
                task_type="basic",
            )
            # Extract answer part (after "A:")
            if "A:" in prompt:
                answer = prompt.split("A:")[1].strip()
                # Check for converted units in answers only
                if "temperature" in answer:
                    # Temperature should be in Celsius
                    if "0.0" in answer or "10.0" in answer or "20.0" in answer:
                        assert (
                            "°C" in answer
                        ), f"Temperature should be converted to Celsius in: {answer}"
                        found_temp = True
                if "pressure" in answer:
                    # Pressure should be in hPa
                    if "1013.25" in answer:
                        assert "hPa" in answer, f"Pressure should be converted to hPa in: {answer}"
                        found_pressure = True
                if "precipitation" in answer:
                    # Precipitation should be in mm
                    if "1.0" in answer:
                        assert (
                            "mm" in answer
                        ), f"Precipitation should be converted to mm in: {answer}"
                        found_precip = True
        # Verify we tested at least some conversions
        assert (
            found_temp or found_pressure or found_precip
        ), "Should have found at least one unit conversion to test"

    def test_mistral_chat_formatting(self, prompt_generator):
        """Test Mistral chat format wrapper."""
        test_prompt = "This is a test prompt."
        # Test without system message
        formatted = prompt_generator.format_for_mistral(test_prompt)
        assert formatted == "<s>[INST] This is a test prompt. [/INST]"
        # Test with system message
        system_msg = "You are a helpful assistant."
        formatted_with_system = prompt_generator.format_for_mistral(test_prompt, system_msg)
        expected = f"<s>[INST] {system_msg}\n\n{test_prompt} [/INST]"
        assert formatted_with_system == expected

    def test_invalid_task_type(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test error handling for invalid task types."""
        with pytest.raises(ValueError, match="Unknown task type"):
            prompt_generator.generate_analysis_prompt(
                location=mock_location,
                statistics=mock_statistics,
                statistics_table=statistics_table,
                task_type="invalid_task",
            )

    def test_context_summary_generation(self, prompt_generator, mock_statistics):
        """Test context summary generation from statistics."""
        context_summary = prompt_generator._build_context_summary(mock_statistics)
        # Should include temperature, wind, pressure, precipitation
        assert "16.9°C" in context_summary  # 290K - 273.15
        assert "2.9 m/s" in context_summary  # wind speed calculation
        assert "1010" in context_summary and "hPa" in context_summary  # pressure conversion
        assert "1.0 mm" in context_summary  # precipitation conversion

    def test_context_summary_empty_statistics(self, prompt_generator):
        """Test context summary with no relevant statistics."""
        empty_stats = [
            VariableStatistics(
                variable_name="irrelevant_var",
                min_value=0.0,
                max_value=1.0,
                mean_value=0.5,
                std_value=1.0,
                unit="unitless",
                description="irrelevant variable",
            )
        ]
        context_summary = prompt_generator._build_context_summary(empty_stats)
        assert context_summary == "various weather conditions"

    def test_extract_key_insights(self, prompt_generator, mock_statistics):
        """Test key insights extraction."""
        insights = prompt_generator.extract_key_insights(mock_statistics)
        # Temperature insight
        assert "temperature" in insights
        assert "Mild" in insights["temperature"]  # 16.9°C should be "Mild"
        # Wind insight
        assert "wind" in insights
        assert "Light winds" in insights["wind"]  # 2.9 m/s should be "Light"
        # Pressure insight
        assert "pressure" in insights
        assert "Normal pressure" in insights["pressure"]  # 1010 hPa is normal

    def test_create_instruction_format(self, prompt_generator, mock_location, statistics_table):
        """Test instruction format creation."""
        instruction = "Analyze the weather data."
        inst, input_data = prompt_generator.create_instruction_format(
            location=mock_location, statistics_table=statistics_table, instruction=instruction
        )
        assert inst == instruction
        assert mock_location.description in input_data
        assert "45.00°N" in input_data
        assert "-120.00°E" in input_data
        assert statistics_table in input_data

    def test_template_rendering_with_missing_variables(
        self, prompt_generator, mock_location, statistics_table
    ):
        """Test template rendering handles missing optional variables gracefully."""
        # Test with no context_summary (should not crash)
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=[],  # Empty statistics
            statistics_table=statistics_table,
            task_type="weather_description",
        )
        # Should still generate a valid prompt
        assert len(prompt) > 0
        assert mock_location.name in prompt
        assert statistics_table in prompt

    def test_template_variable_interpolation(
        self, prompt_generator, mock_location, statistics_table
    ):
        """Test that all template variables are properly interpolated."""
        prompt = prompt_generator.generate_analysis_prompt(
            location=mock_location,
            statistics=[],
            statistics_table=statistics_table,
            task_type="weather_description",
        )
        # Should not contain any unrendered Jinja2 syntax
        assert "{{" not in prompt
        assert "}}" not in prompt
        assert "{%" not in prompt
        assert "%}" not in prompt

    def test_all_task_types_available(
        self, prompt_generator, mock_location, mock_statistics, statistics_table
    ):
        """Test that all documented task types work."""
        task_types = ["weather_description", "forecast", "anomaly_detection"]
        for task_type in task_types:
            prompt = prompt_generator.generate_analysis_prompt(
                location=mock_location,
                statistics=mock_statistics,
                statistics_table=statistics_table,
                task_type=task_type,
            )
            # Each should produce a non-empty prompt
            assert len(prompt) > 100
            assert mock_location.name in prompt
            assert statistics_table in prompt


class TestPromptTemplateFiles:
    """Test the actual template files for syntax and structure."""

    @pytest.fixture
    def templates_dir(self):
        """Get the templates directory."""
        return Path(__file__).parent.parent.parent / "training" / "prompts"

    def test_template_files_exist(self, templates_dir):
        """Test that all required template files exist."""
        required_templates = [
            "weather_description.jinja2",
            "forecast.jinja2",
            "anomaly_detection.jinja2",
            "multimodal_training.jinja2",
            "basic.jinja2",
            "mistral_chat_wrapper.jinja2",
        ]
        for template_file in required_templates:
            template_path = templates_dir / template_file
            assert template_path.exists(), f"Template file {template_file} not found"
            assert template_path.stat().st_size > 0, f"Template file {template_file} is empty"

    def test_template_syntax_valid(self, templates_dir):
        """Test that all templates have valid Jinja2 syntax."""
        from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template_files = list(templates_dir.glob("*.jinja2"))
        assert len(template_files) > 0, "No template files found"
        for template_file in template_files:
            template_name = template_file.name
            try:
                # This will raise TemplateSyntaxError if syntax is invalid
                env.get_template(template_name)
            except TemplateSyntaxError as e:
                pytest.fail(f"Template {template_name} has invalid syntax: {e}")

    def test_template_required_variables(self, templates_dir):
        """Test that templates contain expected variable references."""
        from jinja2 import Environment, FileSystemLoader, meta

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        # Test weather_description template
        template_source = env.loader.get_source(env, "weather_description.jinja2")
        parsed_content = env.parse(template_source)
        variables = meta.find_undeclared_variables(parsed_content)
        # Should reference key variables
        expected_vars = {"location", "statistics_table"}
        assert expected_vars.issubset(
            variables
        ), f"Missing variables in weather_description: {expected_vars - variables}"

    def test_mistral_chat_wrapper_format(self, templates_dir):
        """Test that Mistral chat wrapper produces correct format."""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("mistral_chat_wrapper.jinja2")
        # Test without system message
        result = template.render(prompt="Test prompt")
        assert result == "<s>[INST] Test prompt [/INST]"
        # Test with system message
        result_with_system = template.render(prompt="Test prompt", system_message="System message")
        assert result_with_system == "<s>[INST] System message\n\nTest prompt [/INST]"
