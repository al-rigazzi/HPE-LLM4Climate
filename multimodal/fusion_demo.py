"""
Example usage of multimodal climate-text fusion.

This script demonstrates how to use the ClimateTextFusion model to combine
climate data from PrithviWxC with text processing using Llama 3.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List
import warnings

# Import our multimodal fusion classes
from multimodal.climate_text_fusion import (
    ClimateTextFusion,
    ClimateQuestionAnswering,
    ClimateTextGeneration
)


def create_dummy_climate_batch(batch_size: int = 2, reduced_size: bool = True) -> Dict[str, torch.Tensor]:
    """
    Create dummy climate data for testing.

    Args:
        batch_size: Number of samples in batch
        reduced_size: Whether to use smaller dimensions for memory efficiency

    Returns:
        climate_batch: Dictionary with climate data tensors
    """
    if reduced_size:
        # Use smaller dimensions for demo purposes to avoid memory issues
        n_times = 2
        n_channels = 69  # Match the corrected encoder configuration
        n_static_channels = 4
        n_lats = 36      # Reduced from 720 for demo
        n_lons = 58      # Reduced from 1440 for demo
    else:
        # Original dimensions (might cause memory issues)
        n_times = 2
        n_channels = 160
        n_static_channels = 4
        n_lats = 720
        n_lons = 1440

    return {
        'x': torch.randn(batch_size, n_times, n_channels, n_lats, n_lons),
        'static': torch.randn(batch_size, n_static_channels, n_lats, n_lons),
        'climate': torch.randn(batch_size, n_channels, n_lats, n_lons),
        'input_time': torch.tensor([0.0, 3.0])[:batch_size],
        'lead_time': torch.tensor([18.0, 18.0])[:batch_size],
    }


def example_multimodal_fusion():
    """
    Example 1: Basic multimodal fusion between climate data and text.
    """
    print("=== Example 1: Basic Multimodal Fusion ===\n")

    try:
        # Initialize the fusion model
        # Note: This will use a smaller Llama model for demonstration
        print("Initializing multimodal fusion model...")

        # Try with corrected encoder first
        try:
            fusion_model = ClimateTextFusion(
                prithvi_encoder_path='data/weights/prithvi_encoder_corrected.pt',
                llama_model_name='distilbert-base-uncased',  # Using smaller model for testing
                fusion_mode='cross_attention',
                max_climate_tokens=256,  # Reduce for memory efficiency
                max_text_length=128,
                freeze_prithvi=True,
                freeze_llama=True
            )
            print("✓ Using corrected encoder successfully!")

        except Exception as encoder_error:
            print(f"Note: Corrected encoder failed ({str(encoder_error)[:100]}...)")
            print("Continuing with demo mode - using mock climate features...")

            # Use demo mode with mock features
            demo_mode = True
            print("✓ Demo mode initialized - will simulate fusion!")

        # Create sample data with reduced dimensions for memory efficiency
        climate_batch = create_dummy_climate_batch(batch_size=2, reduced_size=True)
        text_inputs = [
            "How will tornado frequency change by 2050?",
            "What is the best crop to plant in Sweden considering climate projections for 2050?"
        ]

        print(f"Climate data shape: {climate_batch['x'].shape}")
        print(f"Text inputs: {text_inputs}")

        # Perform multimodal fusion
        print("\nPerforming multimodal fusion...")

        if 'fusion_model' in locals():
            # Use real model
            with torch.no_grad():
                outputs = fusion_model(climate_batch, text_inputs)

            print(f"✓ Fusion completed!")
            print(f"  Fused features shape: {outputs['fused_features'].shape}")
            print(f"  Climate features shape: {outputs['climate_features'].shape}")
            print(f"  Text features shape: {outputs['text_features'].shape}")
        else:
            # Simulate fusion for demo
            batch_size = len(text_inputs)
            feature_dim = 512

            mock_outputs = {
                'fused_features': torch.randn(batch_size, feature_dim),
                'climate_features': torch.randn(batch_size, feature_dim // 2),
                'text_features': torch.randn(batch_size, feature_dim // 2)
            }

            print(f"✓ Fusion simulation completed!")
            print(f"  Fused features shape: {mock_outputs['fused_features'].shape}")
            print(f"  Climate features shape: {mock_outputs['climate_features'].shape}")
            print(f"  Text features shape: {mock_outputs['text_features'].shape}")
            print("  Note: These are mock features for demonstration purposes.")

    except Exception as e:
        print(f"✗ Error in multimodal fusion: {e}")
        print("This is expected if running without GPU or with limited memory.")
        print("The remaining demos will show the conceptual framework.")


def example_climate_qa():
    """
    Example 2: Climate-aware question answering.
    """
    print("\n=== Example 2: Climate Question Answering ===\n")

    print("Note: This example shows the structure for climate QA.")
    print("Actual implementation would require:")
    print("- Properly formatted climate data")
    print("- Real climate questions and answers")
    print("- Training on climate QA datasets")

    # Pseudo-code for climate QA
    sample_questions = [
        "How much more likely will tornadoes be in 2050 compared to now?",
        "What is the long-term climate trend for this region?",
        "How sustainable will agriculture be in this area by 2100?"
    ]

    print(f"Sample questions: {sample_questions}")
    print("QA model would output probability scores for each question.")


def example_climate_text_generation():
    """
    Example 3: Climate-conditioned text generation.
    """
    print("\n=== Example 3: Climate Text Generation ===\n")

    print("Climate text generation would enable:")
    print("- Automated climate assessment generation")
    print("- Climate trend summarization")
    print("- Scientific climate impact report writing")

    sample_outputs = [
        "Based on climate model projections and historical trends, tornado frequency is expected to increase by 15-25% by 2050...",
        "Long-term climate analysis indicates significant changes in agricultural viability for northern European regions...",
        "Regional sustainability assessments show that southwestern US regions will face increasing challenges by 2100..."
    ]

    print("Sample generated texts:")
    for i, text in enumerate(sample_outputs, 1):
        print(f"  {i}. {text}")


def demonstrate_fusion_modes():
    """
    Demonstrate different fusion strategies.
    """
    print("\n=== Fusion Mode Comparison ===\n")

    fusion_modes = {
        'cross_attention': "Uses attention mechanism to align climate and text features",
        'concatenate': "Simply concatenates climate and text features",
        'add': "Projects climate features to text space and adds them"
    }

    print("Available fusion modes:")
    for mode, description in fusion_modes.items():
        print(f"  • {mode}: {description}")

    print("\nRecommendations:")
    print("  • Use 'cross_attention' for complex reasoning tasks")
    print("  • Use 'concatenate' for simple feature combination")
    print("  • Use 'add' for lightweight fusion with limited compute")


def show_practical_applications():
    """
    Show practical applications of climate-text fusion.
    """
    print("\n=== Practical Applications ===\n")

    applications = {
        "Climate Trend Analysis": [
            "Generate long-term climate trend reports",
            "Answer questions about future climate patterns",
            "Explain climate projection uncertainties"
        ],
        "Climate Research": [
            "Analyze climate pattern descriptions",
            "Generate research summaries",
            "Compare model predictions with observations"
        ],
        "Agriculture": [
            "Provide long-term crop viability assessments",
            "Generate climate-informed agricultural planning",
            "Predict future growing season conditions"
        ],
        "Regional Planning": [
            "Generate climate risk assessments",
            "Assess regional sustainability from climate data",
            "Create long-term adaptation recommendations"
        ],
        "Education": [
            "Explain climate change phenomena in simple terms",
            "Generate educational climate content",
            "Answer student questions about climate science"
        ]
    }

    for application, use_cases in applications.items():
        print(f"**{application}:**")
        for use_case in use_cases:
            print(f"  - {use_case}")
        print()


def performance_considerations():
    """
    Discuss performance and optimization considerations.
    """
    print("=== Performance Considerations ===\n")

    considerations = {
        "Memory Requirements": [
            "PrithviWxC encoder: ~8GB GPU memory",
            "Llama 3 model: 6-12GB depending on size",
            "Consider using smaller models for development"
        ],
        "Computational Efficiency": [
            "Freeze pre-trained models during initial experiments",
            "Use gradient checkpointing for memory efficiency",
            "Batch processing for multiple samples"
        ],
        "Model Selection": [
            "Start with smaller Llama models (1B-3B parameters)",
            "Use DistilBERT or similar for initial prototyping",
            "Scale up once architecture is validated"
        ],
        "Data Preprocessing": [
            "Limit climate token count for memory efficiency",
            "Precompute climate features when possible",
            "Cache tokenized text inputs"
        ]
    }

    for category, items in considerations.items():
        print(f"**{category}:**")
        for item in items:
            print(f"  • {item}")
        print()


def main():
    """
    Main function demonstrating multimodal climate-text fusion.
    """
    print("🌍 Climate-Text Multimodal Fusion Demo 🤖\n")
    print("This demo shows how to combine climate data with text using AI.\n")

    # Run examples
    example_multimodal_fusion()
    example_climate_qa()
    example_climate_text_generation()
    demonstrate_fusion_modes()
    show_practical_applications()
    performance_considerations()

    print("=== Next Steps ===\n")
    print("To use this system with real data:")
    print("1. Prepare your climate datasets in the expected format")
    print("2. Choose appropriate text data (weather reports, research papers, etc.)")
    print("3. Fine-tune the fusion model on your specific task")
    print("4. Evaluate performance on held-out test data")
    print("5. Deploy for real-world applications")

    print("\n✨ Demo completed! Check the code for implementation details.")


if __name__ == "__main__":
    main()
