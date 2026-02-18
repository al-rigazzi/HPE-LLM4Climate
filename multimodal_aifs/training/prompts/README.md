# Climate Prompt Templates

This directory contains Jinja2 templates for generating climate analysis prompts. These templates separate the prompt structure from the Python code, making it easy to modify prompt content without changing code.

## Template Files

### Core Templates

- **`weather_description.jinja2`** - Detailed weather condition analysis prompts
- **`forecast.jinja2`** - Short-term forecasting analysis prompts
- **`anomaly_detection.jinja2`** - Atmospheric anomaly detection prompts
- **`rl_training.jinja2`** - Numeric-only output for RL training stage

### Formatting Templates

- **`mistral_chat_wrapper.jinja2`** - Wraps prompts in Mistral's `[INST]` chat format

## Template Variables

All templates have access to these variables:

### Location Object (`location`)
- `location.name` - Location name (e.g., "Australia", "Pacific Ocean")
- `location.location_type` - Type (e.g., "continent", "body_of_water", "country")
- `location.description` - Human-readable description
- `location.center_lat` - Latitude coordinate (float)
- `location.center_lon` - Longitude coordinate (float)

### Statistics
- `statistics_table` - Pre-formatted table of climate variable statistics
- `statistics` - List of VariableStatistics objects (for custom processing)
- `context_summary` - Human-readable weather summary (weather_description only)

## Example Usage

```python
from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator

# Initialize with default templates directory
prompt_gen = ClimatePromptGenerator()

# Generate different types of prompts
weather_prompt = prompt_gen.generate_analysis_prompt(
    location=location,
    statistics=stats,
    statistics_table=table,
    task_type="weather_description"
)

forecast_prompt = prompt_gen.generate_analysis_prompt(
    location=location,
    statistics=stats,
    statistics_table=table,
    task_type="forecast"
)

# Format for Mistral
formatted = prompt_gen.format_for_mistral(weather_prompt)
```

## Template Syntax

Templates use Jinja2 syntax:

```jinja2
{# Comments #}
{{ variable }} {# Variable interpolation #}
{% if condition %} ... {% endif %} {# Conditionals #}
{% for item in list %} ... {% endfor %} {# Loops #}
{{ "%.2f"|format(number) }} {# Filters #}
```

## Customization

To modify prompts:

1. **Edit existing templates** - Change wording, structure, instructions
2. **Add new templates** - Create new `.jinja2` files and update the generator
3. **Custom template directory** - Pass `templates_dir` to `ClimatePromptGenerator()`

## Template Design Principles

- **Clean separation** - Templates contain text, Python contains logic
- **Maintainable** - Easy for non-developers to modify prompt content
- **Flexible** - Support conditionals and loops for complex prompts
- **Versioned** - Template changes are tracked in git like code
- **Production ready** - Industry standard templating approach