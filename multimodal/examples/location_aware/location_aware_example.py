"""
Location-Aware Climate Analysis Example

This example demonstrates how to use the location-aware climate analysis system
to answer geographic-specific climate questions with spatial attention masking.

Example Usage:
    python location_aware_example.py

Features Demonstrated:
- Geographic entity resolution from natural language
- Spatial attention masking for specific regions
- Location-aware multimodal fusion
- Climate risk assessment with geographic context
- Visualization of attention patterns
"""

import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    from .climate_text_fusion import ClimateTextFusion
    from .location_aware import (
        GeographicResolver,
        SpatialCropper,
        demo_location_resolution,
        demo_spatial_masking,
    )
    from .location_aware_fusion import FusionMode, LocationAwareClimateAnalysis
except ImportError:
    from climate_text_fusion import ClimateTextFusion
    from location_aware import (
        GeographicResolver,
        SpatialCropper,
        demo_location_resolution,
        demo_spatial_masking,
    )
    from location_aware_fusion import FusionMode, LocationAwareClimateAnalysis


def create_mock_climate_data(grid_shape: tuple = (36, 58)) -> torch.Tensor:
    """
    Create realistic mock climate data that mimics Prithvi-WxC output.

    Note: Using smaller grid for demo purposes to reduce memory usage.

    Args:
        grid_shape: (n_lats, n_lons) grid dimensions (reduced for demo)

    Returns:
        Climate features tensor [1, seq_len, feature_dim]
    """
    n_lats, n_lons = grid_shape
    n_patches = (n_lats // 2) * (n_lons // 2)  # 2x2 patch size
    feature_dim = 128  # Reduced dimension for demo

    # Create spatially correlated climate features
    # Simulate temperature, precipitation, and pressure patterns

    # Temperature: warmer at equator, colder at poles
    lat_indices = torch.arange(n_lats // 2).float()
    lat_temp_pattern = torch.cos((lat_indices - n_lats // 4) * np.pi / (n_lats // 2))

    # Precipitation: higher in tropics and mid-latitudes
    lat_precip_pattern = torch.exp(
        -(((lat_indices - n_lats // 4) / (n_lats // 8)) ** 2)
    ) + 0.5 * torch.exp(-(((lat_indices - n_lats // 6) / (n_lats // 12)) ** 2))

    # Create features for each patch
    features = []
    patch_idx = 0

    for i in range(n_lats // 2):
        for j in range(n_lons // 2):
            # Base climate features
            temp_base = lat_temp_pattern[i] + 0.1 * torch.randn(1)
            precip_base = lat_precip_pattern[i] + 0.1 * torch.randn(1)

            # Add longitude variations (ocean vs land patterns)
            lon_factor = torch.sin(torch.tensor(2 * np.pi * j / (n_lons // 2)))

            # Create feature vector (reduced size for demo)
            patch_features = torch.cat(
                [
                    torch.full((32,), temp_base.item()),  # Temperature features
                    torch.full((32,), precip_base.item()),  # Precipitation features
                    torch.full((64,), lon_factor.item()),  # Geographic features
                ]
            )

            features.append(patch_features)
            patch_idx += 1

    # Stack and add batch dimension
    climate_tensor = torch.stack(features).unsqueeze(0)  # [1, n_patches, 768]

    return climate_tensor


def visualize_attention_pattern(
    spatial_mask: torch.Tensor,
    location_info: Optional[Dict],
    title: str = "Spatial Attention Pattern",
):
    """
    Visualize spatial attention mask on world map.

    Args:
        spatial_mask: Attention mask [n_lats, n_lons]
        location_info: Geographic location information
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Create latitude and longitude grids
        n_lats, n_lons = spatial_mask.shape
        lats = np.linspace(-90, 90, n_lats)
        lons = np.linspace(-180, 180, n_lons)

        # Plot attention mask
        im = ax.imshow(
            spatial_mask.numpy(),
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap="hot",
            alpha=0.7,
        )

        # Add geographic context
        if location_info:
            bounds = location_info["bounds"]
            rect = plt.Rectangle(
                (bounds["lon_min"], bounds["lat_min"]),
                bounds["lon_max"] - bounds["lon_min"],
                bounds["lat_max"] - bounds["lat_min"],
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add location label
            center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
            center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
            ax.plot(center_lon, center_lat, "c*", markersize=15)
            ax.text(
                center_lon,
                center_lat + 5,
                location_info["name"],
                ha="center",
                va="bottom",
                color="cyan",
                fontweight="bold",
            )

        # Formatting
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")


def analyze_climate_question(
    model: LocationAwareClimateAnalysis,
    climate_data: torch.Tensor,
    question: str,
    verbose: bool = True,
) -> Dict:
    """
    Analyze a climate question with location awareness.

    Args:
        model: Location-aware climate analysis model
        climate_data: Climate features tensor
        question: Natural language climate question
        verbose: Whether to print detailed results

    Returns:
        Analysis results dictionary
    """
    if verbose:
        print(f"🌍 Analyzing: {question}")
        print("-" * 60)

    # Analyze the question
    with torch.no_grad():
        result = model.analyze_location_query(climate_data, question, return_visualization=True)

    if verbose:
        print(f"📍 Location: {result['location']} ({result['location_type']})")
        print(f"⚠️  Climate Risk: {result['climate_risk']}")
        print(f"📊 Risk Confidence: {result['risk_confidence']:.1%}")
        print(f"📈 Trend Magnitude: {result['trend_magnitude']:.2f}")
        print(f"🎯 Overall Confidence: {result['overall_confidence']:.1%}")
        print(f"\n💭 Interpretation:")
        print(result["interpretation"])

        # Show attention statistics
        if result.get("attention_weights") is not None:
            attention = result["attention_weights"]
            print(f"\n🔍 Attention Analysis:")
            print(f"   Max attention: {attention.max():.3f}")
            print(f"   Mean attention: {attention.mean():.3f}")
            print(f"   Focused pixels: {(attention > attention.mean() + attention.std()).sum()}")

        print("\n" + "=" * 70 + "\n")

    return result


def compare_fusion_modes():
    """Compare different fusion modes for location-aware analysis."""
    print("🔄 Comparing Fusion Modes for Location-Aware Analysis\n")

    # Create model and data
    model = LocationAwareClimateAnalysis()
    model.eval()
    climate_data = create_mock_climate_data()

    question = "What crops will be viable in Sweden by 2050?"
    fusion_modes = [
        FusionMode.CONCATENATION,
        FusionMode.CROSS_ATTENTION,
        FusionMode.ADDITIVE,
    ]

    results = {}

    with torch.no_grad():
        for mode in fusion_modes:
            print(f"Testing fusion mode: {mode}")

            # Forward pass with specific fusion mode
            result = model.forward(climate_data, question, fusion_mode=mode)

            # Extract key metrics
            risk_probs = F.softmax(result["climate_risk"], dim=-1)
            risk_categories = ["Low Risk", "Moderate Risk", "High Risk"]
            predicted_risk = risk_categories[risk_probs.argmax(dim=-1).item()]
            confidence = result["confidence"].item()

            results[mode] = {
                "risk": predicted_risk,
                "confidence": confidence,
                "risk_probs": risk_probs.squeeze().tolist(),
            }

            print(f"  Risk: {predicted_risk}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Risk Distribution: {[f'{p:.3f}' for p in risk_probs.squeeze().tolist()]}")
            print()

    # Compare results
    print("📊 Fusion Mode Comparison Summary:")
    for mode, result in results.items():
        print(f"  {mode}: {result['risk']} (conf: {result['confidence']:.3f})")


def demonstrate_geographic_coverage():
    """Demonstrate analysis across different geographic regions."""
    print("🗺️  Geographic Coverage Demonstration\n")

    model = LocationAwareClimateAnalysis()
    model.eval()
    climate_data = create_mock_climate_data()

    # Questions covering different regions and scales
    geographic_questions = [
        ("Point Location", "Climate risks at 59.3°N, 18.1°E by 2080?"),
        ("City/Country", "How will drought affect agriculture in Sweden?"),
        ("State/Province", "Wildfire risk changes in California by 2060?"),
        ("Large Region", "Arctic ice melting acceleration patterns?"),
        ("Global Context", "Ocean temperature rise trends worldwide?"),
    ]

    coverage_results = []

    for category, question in geographic_questions:
        print(f"🌍 {category}")
        result = analyze_climate_question(model, climate_data, question, verbose=False)

        coverage_results.append(
            {
                "category": category,
                "question": question,
                "location": result["location"],
                "location_type": result["location_type"],
                "risk": result["climate_risk"],
                "confidence": result["overall_confidence"],
            }
        )

        print(f"   Question: {question}")
        print(f"   Location: {result['location']} ({result['location_type']})")
        print(
            f"   Assessment: {result['climate_risk']} (confidence: {result['overall_confidence']:.1%})"
        )
        print()

    # Summary
    print("📋 Coverage Summary:")
    location_types = {}
    for result in coverage_results:
        loc_type = result["location_type"]
        if loc_type not in location_types:
            location_types[loc_type] = 0
        location_types[loc_type] += 1

    for loc_type, count in location_types.items():
        print(f"   {loc_type.title()}: {count} questions")


def main():
    """Main demonstration of location-aware climate analysis."""
    print("🌍 Location-Aware Climate Analysis System")
    print("=" * 50)
    print()

    # 1. Basic geographic resolution demo
    print("Step 1: Geographic Resolution Capabilities")
    demo_location_resolution()
    print()

    # 2. Spatial masking demo
    print("Step 2: Spatial Attention Masking")
    demo_spatial_masking()
    print()

    # 3. Create model and mock data
    print("Step 3: Location-Aware Climate Analysis")
    try:
        print("🔄 Initializing location-aware climate model...")
        model = LocationAwareClimateAnalysis()
        print("✅ Model loaded successfully!")
        model.eval()
        climate_data = create_mock_climate_data()

        # 4. Analyze specific questions
        example_questions = [
            "What crops will be viable in Sweden by 2050?",
            "How will tornado frequency change in Texas?",
            "Climate resilience planning for Mediterranean agriculture",
            "Arctic permafrost melting acceleration",
        ]

        for question in example_questions:
            result = analyze_climate_question(model, climate_data, question)

            # Visualize attention pattern for interesting cases
            if result.get("spatial_mask") is not None and result.get("location_info"):
                print(f"🎨 Visualizing attention pattern for: {result['location']}")
                try:
                    visualize_attention_pattern(
                        result["spatial_mask"].squeeze(0),
                        result["location_info"],
                        f"Spatial Attention: {result['location']}",
                    )
                except Exception as e:
                    print(f"Visualization skipped: {e}")

    except (MemoryError, RuntimeError, OSError) as e:
        print(f"⚠️  Full model demonstration skipped due to resource constraints:")
        print(f"   {str(e)}")
        print("\n💡 This demo requires significant computational resources.")
        print("   For full functionality, ensure you have:")
        print("   - Sufficient RAM (8GB+ recommended)")
        print("   - Access to Hugging Face models (authentication may be required)")
        print("   - Or provide pre-trained model weights")
        print("\n✅ Steps 1 & 2 (Geographic Resolution & Spatial Masking) completed successfully!")
        print("   These demonstrate the core location-aware capabilities.")

        # Demonstrate just the location processing without heavy models
        print("\n🌍 Mini Demo - Location Processing Only:")
        resolver = GeographicResolver()
        cropper = SpatialCropper()

        for question in [
            "What crops will be viable in Sweden by 2050?",
            "Arctic ice melting patterns",
        ]:
            print(f"\n📝 Processing: {question}")
            locations = resolver.extract_locations(question)
            if locations:
                location = resolver.resolve_location(locations[0])
                if location:
                    mask = cropper.create_location_mask(location, "gaussian")
                    print(f"   ✓ Location found: {location.name} ({location.location_type})")
                    print(f"   ✓ Spatial mask created: {mask.sum():.0f} focused pixels")
                else:
                    print(f"   ✗ Could not resolve location: {locations[0]}")
            else:
                print(f"   ✗ No locations found in query")
    except Exception as e:
        print(f"⚠️  Full model demonstration skipped due to unexpected error:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\n✅ Steps 1 & 2 (Geographic Resolution & Spatial Masking) completed successfully!")
        print(
            "   These demonstrate the core location-aware capabilities."
        )  # 5. Compare fusion modes
    compare_fusion_modes()

    # 6. Demonstrate geographic coverage
    demonstrate_geographic_coverage()

    print("✅ Location-aware climate analysis demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("- Geographic entity resolution from natural language")
    print("- Spatial attention masking for regional focus")
    print("- Location-aware multimodal fusion")
    print("- Climate risk assessment with geographic context")
    print("- Multiple fusion strategies")
    print("- Global to local scale analysis")


if __name__ == "__main__":
    main()
