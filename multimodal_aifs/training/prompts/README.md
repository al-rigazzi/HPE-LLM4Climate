# Prompt Templates

Jinja2 templates used by `multimodal_aifs/training/prompt_generator.py`.

## Current Templates

- `weather_description.jinja2`
- `forecast.jinja2`
- `anomaly_detection.jinja2`
- `basic.jinja2`
- `rl_training.jinja2`
- `mistral_chat_wrapper.jinja2`

## Supported Task Types

`ClimatePromptGenerator.generate_analysis_prompt(..., task_type=...)` supports:

- `weather_description`
- `forecast`
- `anomaly_detection`
- `basic`

RL-specific prompt generation uses `generate_rl_training_prompt(...)` with `rl_training.jinja2`.

## Variables Available to Templates

- `location`
- `statistics`
- `statistics_table`
- `context_summary` (weather description flow)

## Quick Usage

```python
from multimodal_aifs.training.prompt_generator import ClimatePromptGenerator

generator = ClimatePromptGenerator()
prompt = generator.generate_analysis_prompt(
    location=location,
    statistics=statistics,
    statistics_table=statistics_table,
    task_type="weather_description",
)

formatted = generator.format_for_mistral(prompt)
```
