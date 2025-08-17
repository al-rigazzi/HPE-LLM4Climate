# Examples Directory

This directory contains comprehensive examples demonstrating the multimodal climate analysis system.

## Directory Structure

### `basic/`
Basic examples and simple demonstrations:
- **`example_usage.py`** - Basic usage patterns and simple examples
- **`practical_example.py`** - Practical implementation examples
- **`simple_prithvi_demo.py`** - Simple PrithviWxC encoder demonstration

### `advanced/`
Advanced examples with complex functionality:
- **`complete_real_weights_demo.py`** - Complete demonstration with real model weights
- **`fusion_demo.py`** - Comprehensive multimodal fusion demonstration
- **`public_model_demo.py`** - Public model integration examples
- **`real_prithvi_demo.py`** - Real PrithviWxC model integration
- **`working_demo.py`** - Working demonstration of full system

### `location_aware/`
Location-aware climate analysis examples:
- **`final_location_aware_fusion.py`** - Complete location-aware fusion system
- **`location_aware_example.py`** - Location-aware processing examples

## Usage

Each subdirectory contains examples that build upon each other:

1. **Start with `basic/`** - Learn fundamental concepts and simple usage
2. **Progress to `advanced/`** - Explore complex multimodal fusion capabilities
3. **Explore `location_aware/`** - See geographic integration and spatial analysis

For the complete system integration, see the examples in `advanced/` and `location_aware/` which demonstrate the full multimodal climate analysis capabilities.

## Requirements

Most examples require:
- PrithviWxC model weights in `data/weights/`
- HuggingFace authentication for gated models (Meta Llama)
- Internet connectivity for geographic services (location-aware examples)
