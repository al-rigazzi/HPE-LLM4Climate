#!/usr/bin/env python3
"""
AIFS Multimodal Time Series Example

This example demonstrates the complete multimodal time series workflow using
AIFS encoder for spatial-temporal climate data tokenization, combined with
text descriptions for comprehensive climate analysis.

Features demonstrated:
- 5-D time series data tokenization using AIFSTimeSeriesTokenizer
- Multimodal fusion of climate time series with text descriptions
- Different temporal modeling approaches (transformer, LSTM, none)
- Real-world use cases and applications
- Performance comparison across strategies

Usage:
    python multimodal_aifs/examples/multimodal_timeseries_demo.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion
from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer


def create_synthetic_timeseries_data(
    batch_size: int = 4, time_steps: int = 8, n_variables: int = 5, spatial_shape: tuple = (64, 64)
) -> torch.Tensor:
    """
    Create synthetic 5-D climate time series data.

    Args:
        batch_size: Number of samples in batch
        time_steps: Number of temporal steps
        n_variables: Number of climate variables
        spatial_shape: Spatial dimensions (height, width)

    Returns:
        5-D tensor [batch, time, variables, height, width]
    """
    height, width = spatial_shape

    # Create base spatial patterns
    x = torch.linspace(-1, 1, width).unsqueeze(0).repeat(height, 1)
    y = torch.linspace(-1, 1, height).unsqueeze(1).repeat(1, width)

    # Initialize data tensor
    data = torch.zeros(batch_size, time_steps, n_variables, height, width)

    for b in range(batch_size):
        for t in range(time_steps):
            # Convert t to tensor for torch operations
            t_tensor = torch.tensor(t, dtype=torch.float32)

            for v in range(n_variables):
                # Create different patterns for different variables
                if v == 0:  # Temperature-like pattern
                    pattern = torch.sin(x * 3 + t_tensor * 0.5) * torch.cos(y * 2)
                elif v == 1:  # Pressure-like pattern
                    pattern = torch.exp(-(x**2 + y**2) * 0.5) * torch.sin(t_tensor * 0.3)
                elif v == 2:  # Wind-like pattern
                    pattern = x * torch.sin(y * 4 + t_tensor * 0.7)
                elif v == 3:  # Humidity-like pattern
                    pattern = torch.cos(x * y + t_tensor * 0.4)
                else:  # Precipitation-like pattern
                    pattern = torch.relu(torch.sin(x * 2) + torch.cos(y * 3) - 1.5 + t_tensor * 0.1)

                # Add temporal evolution and noise
                temporal_factor = 1 + 0.3 * torch.sin(t_tensor * 0.8)
                noise = torch.randn_like(pattern) * 0.1

                data[b, t, v] = pattern * temporal_factor + noise

    return data


def create_climate_descriptions() -> List[str]:
    """Create realistic climate descriptions for multimodal fusion."""
    descriptions = [
        "High pressure system bringing clear skies and stable temperatures across the region",
        "Low pressure approaching from the west with increasing cloud cover and precipitation",
        "Strong temperature gradient with warm air mass moving northward",
        "Tropical cyclone formation with intense precipitation and high wind speeds",
        "Arctic air mass intrusion causing significant temperature drop",
        "Monsoon pattern with heavy rainfall and high humidity levels",
        "Drought conditions with elevated temperatures and minimal precipitation",
        "Coastal fog formation due to temperature differential over ocean",
    ]
    return descriptions


class MultimodalTimeSeriesDemo:
    """Demonstration class for multimodal time series analysis."""

    def __init__(self, device: str = "cpu"):
        """Initialize the demo with device configuration."""
        self.device = device
        self.tokenizers = {}
        self.fusion_models = {}

    def setup_tokenizers(self):
        """Initialize tokenizers with different temporal modeling approaches."""
        print("🔧 Setting up AIFSTimeSeriesTokenizers...")

        temporal_models = ["transformer", "lstm", "none"]

        for model_type in temporal_models:
            print(f"   Initializing {model_type} tokenizer...")
            try:
                self.tokenizers[model_type] = AIFSTimeSeriesTokenizer(
                    temporal_modeling=model_type, hidden_dim=512, device=self.device
                )
                print(f"   ✅ {model_type.capitalize()} tokenizer ready")
            except Exception as e:
                print(f"   ❌ Failed to initialize {model_type} tokenizer: {e}")

        print(f"   📊 Loaded {len(self.tokenizers)} tokenizers")
        print()

    def demonstrate_tokenization(self):
        """Demonstrate time series tokenization with different approaches."""
        print("🎯 Time Series Tokenization Demonstration")
        print("=" * 60)

        # Create sample 5-D time series data
        print("📊 Creating sample time series data...")

        # Different scales for testing
        test_cases = [
            ("Small", 2, 4, 3, (32, 32)),
            ("Medium", 1, 8, 5, (64, 64)),
            ("Large", 4, 12, 7, (128, 128)),
        ]

        for case_name, batch_size, time_steps, n_vars, spatial_shape in test_cases:
            print(f"\\n🔍 Test Case: {case_name}")
            print(
                f"   Shape: [batch={batch_size}, time={time_steps}, vars={n_vars}, spatial={spatial_shape}]"
            )

            # Create data
            data_5d = create_synthetic_timeseries_data(
                batch_size=batch_size,
                time_steps=time_steps,
                n_variables=n_vars,
                spatial_shape=spatial_shape,
            )

            print(f"   ✅ Generated data: {data_5d.shape}")
            print(f"   📈 Data range: [{data_5d.min():.3f}, {data_5d.max():.3f}]")

            # Test each tokenizer
            for model_type, tokenizer in self.tokenizers.items():
                try:
                    start_time = time.time()
                    tokens = tokenizer.tokenize_time_series(data_5d)
                    tokenization_time = time.time() - start_time

                    print(
                        f"   🔄 {model_type.upper():11s}: {data_5d.shape} -> {tokens.shape} ({tokenization_time:.4f}s)"
                    )

                    # Memory usage estimation
                    input_size = data_5d.numel() * 4  # float32
                    output_size = tokens.numel() * 4  # float32
                    compression_ratio = input_size / output_size

                    print(
                        f"      💾 Compression: {compression_ratio:.1f}x ({input_size/1024:.1f}KB -> {output_size/1024:.1f}KB)"
                    )

                except Exception as e:
                    print(f"   ❌ {model_type.upper():11s}: Failed - {e}")

        print()

    def demonstrate_multimodal_fusion(self):
        """Demonstrate fusion of time series tokens with text descriptions."""
        print("🔗 Multimodal Fusion Demonstration")
        print("=" * 60)

        # Create sample data
        batch_size = 3
        time_steps = 6
        data_5d = create_synthetic_timeseries_data(
            batch_size=batch_size, time_steps=time_steps, n_variables=5, spatial_shape=(64, 64)
        )

        descriptions = create_climate_descriptions()[:batch_size]

        print(f"📊 Sample data: {data_5d.shape}")
        print(f"📝 Text descriptions:")
        for i, desc in enumerate(descriptions):
            print(f"   {i+1}. {desc}")
        print()

        # Test with transformer tokenizer (default)
        if "transformer" in self.tokenizers:
            print("🚀 Using Transformer Tokenizer for Fusion...")

            tokenizer = self.tokenizers["transformer"]

            # Tokenize time series
            climate_tokens = tokenizer.tokenize_time_series(data_5d)
            print(f"   🌍 Climate tokens: {climate_tokens.shape}")

            # Create simple text embeddings (mock)
            text_tokens = torch.randn(batch_size, 768)  # Typical text embedding size
            print(f"   📝 Text tokens: {text_tokens.shape}")

            # Demonstrate attention-based fusion
            print("   🔗 Computing cross-attention between climate and text...")

            # Simple attention mechanism for demonstration
            climate_flat = climate_tokens.mean(dim=1)  # Average over time: [batch, 512]

            # Project to same dimension for similarity calculation
            if climate_flat.shape[1] != text_tokens.shape[1]:
                # Simple linear projection for demo purposes
                projection = nn.Linear(climate_flat.shape[1], text_tokens.shape[1])
                climate_projected = projection(climate_flat)
            else:
                climate_projected = climate_flat

            # Calculate similarity between projected climate and text features
            similarity = torch.cosine_similarity(climate_projected, text_tokens, dim=1)

            print(f"   📊 Climate-Text Similarity Vector: {similarity.shape}")
            print(f"      Sample similarities: {similarity.tolist()}")
            print(f"      Average similarity: {similarity.mean():.4f}")
            print(f"      Max similarity: {similarity.max():.4f}")
            print(f"      Min similarity: {similarity.min():.4f}")

        print()

    def demonstrate_real_world_applications(self):
        """Show real-world application scenarios."""
        print("🌍 Real-World Application Scenarios")
        print("=" * 60)

        applications = [
            {
                "name": "Weather Forecasting",
                "description": "Multi-timestep weather prediction with textual context",
                "data_shape": "(1, 24, 10, 256, 256)",  # 24 hours, 10 variables
                "use_case": "Process hourly weather data to predict next 24 hours",
                "temporal_model": "transformer",
            },
            {
                "name": "Climate Anomaly Detection",
                "description": "Identify unusual climate patterns over time",
                "data_shape": "(32, 168, 8, 128, 128)",  # Week of hourly data
                "use_case": "Detect anomalous patterns in historical climate data",
                "temporal_model": "transformer",
            },
            {
                "name": "Extreme Event Analysis",
                "description": "Analyze development of extreme weather events",
                "data_shape": "(8, 72, 15, 512, 512)",  # 3 days of high-res data
                "use_case": "Track hurricane formation and intensification",
                "temporal_model": "lstm",
            },
            {
                "name": "Climate Change Assessment",
                "description": "Long-term trend analysis with explanatory text",
                "data_shape": "(1, 8760, 5, 64, 64)",  # Yearly hourly data
                "use_case": "Assess multi-year climate trends and attribution",
                "temporal_model": "transformer",
            },
        ]

        for i, app in enumerate(applications, 1):
            print(f"\\n{i}. 🎯 {app['name']}")
            print(f"   📝 {app['description']}")
            print(f"   📊 Data shape: {app['data_shape']}")
            print(f"   🔧 Recommended temporal model: {app['temporal_model']}")
            print(f"   💡 Use case: {app['use_case']}")

            if app["temporal_model"] in self.tokenizers:
                print(f"   ✅ Tokenizer available and ready")
            else:
                print(f"   ❌ Tokenizer not available")

        print()

        # Performance recommendations
        print("⚡ Performance Recommendations:")
        print("   • Transformer: Best for attention-based temporal relationships")
        print("   • LSTM: Good for sequential dependencies, memory efficient")
        print("   • None: Fastest for spatial-only analysis")
        print("   • Use chunking for very long sequences (>100 timesteps)")
        print("   • Consider downsampling high-resolution spatial data if needed")
        print()

    def run_performance_benchmark(self):
        """Benchmark different tokenizer configurations."""
        print("⚡ Performance Benchmark")
        print("=" * 60)

        # Test configurations
        configs = [
            ("Small Sequence", 4, 8, 5, (32, 32)),
            ("Medium Sequence", 2, 16, 7, (64, 64)),
            ("Large Spatial", 1, 8, 5, (128, 128)),
        ]

        print(
            f"{'Configuration':<20} {'Model':<12} {'Time (s)':<10} {'Throughput':<15} {'Memory':<10}"
        )
        print("-" * 75)

        for config_name, batch_size, time_steps, n_vars, spatial_shape in configs:
            # Create test data
            data = create_synthetic_timeseries_data(
                batch_size=batch_size,
                time_steps=time_steps,
                n_variables=n_vars,
                spatial_shape=spatial_shape,
            )

            for model_type, tokenizer in self.tokenizers.items():
                try:
                    # Warmup
                    _ = tokenizer.tokenize_time_series(data)

                    # Benchmark
                    start_time = time.time()
                    tokens = tokenizer.tokenize_time_series(data)
                    end_time = time.time()

                    processing_time = end_time - start_time
                    throughput = batch_size / processing_time
                    memory_mb = tokens.numel() * 4 / (1024 * 1024)  # MB

                    print(
                        f"{config_name:<20} {model_type:<12} {processing_time:<10.4f} {throughput:<15.1f} {memory_mb:<10.2f}"
                    )

                except Exception as e:
                    print(f"{config_name:<20} {model_type:<12} ERROR: {str(e)[:30]}")

        print()

    def run_demo(self):
        """Run the complete demonstration."""
        print("🌍 AIFS Multimodal Time Series Demonstration")
        print("=" * 80)
        print("This demo showcases the AIFSTimeSeriesTokenizer for climate data analysis")
        print()

        # Setup
        self.setup_tokenizers()

        if not self.tokenizers:
            print("❌ No tokenizers available. Cannot continue demo.")
            return

        # Run demonstrations
        self.demonstrate_tokenization()
        self.demonstrate_multimodal_fusion()
        self.demonstrate_real_world_applications()
        self.run_performance_benchmark()

        # Summary
        print("🎯 Demo Summary")
        print("=" * 40)
        print(f"✅ Tokenizers tested: {list(self.tokenizers.keys())}")
        print(f"✅ 5-D tensor processing demonstrated")
        print(f"✅ Multimodal fusion showcased")
        print(f"✅ Real-world applications outlined")
        print(f"✅ Performance benchmarks completed")
        print()
        print("🚀 Next Steps:")
        print("   • Integrate with your climate datasets")
        print("   • Experiment with different temporal modeling approaches")
        print("   • Scale to larger spatial and temporal resolutions")
        print("   • Combine with advanced text embedding models")
        print("   • Deploy for real-time climate analysis")
        print()
        print("✨ Ready for production climate AI applications!")


def main():
    """Main demonstration entry point."""
    try:
        demo = MultimodalTimeSeriesDemo(device="cpu")
        demo.run_demo()
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
