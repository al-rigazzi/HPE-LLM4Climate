#!/usr/bin/env python3
"""
Inference script for trained AIFS multimodal climate-text fusion models.

This script loads a trained AIFS-based model and performs inference on climate data and text queries.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateTextFusion


class AIFSMultimodalInference:
    """Inference engine for trained multimodal models."""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)

        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load configuration
        config_path = self.checkpoint_path / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Initialize model
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        print("Model loaded successfully!")

    def _load_model(self) -> AIFSClimateTextFusion:
        """Load the trained AIFS model."""
        # Initialize model with same config used during training
        model = AIFSClimateTextFusion(
            aifs_encoder_path=self.config["model"]["aifs_encoder_path"],
            climate_dim=self.config["model"].get("climate_dim", 1024),
            text_dim=self.config["model"].get("text_dim", 768),
            fusion_dim=self.config["model"].get("fusion_dim", 512),
            num_attention_heads=self.config["model"].get("num_fusion_layers", 4),
            dropout=self.config["model"].get("dropout", 0.1),
            device=str(self.device),
        )

        # Load model weights
        # Note: For DeepSpeed checkpoints, you might need different loading logic
        model_path = self.checkpoint_path / "mp_rank_00_model_states.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: Model weights not found, using randomly initialized model")

        model.to(self.device)
        model.eval()

        return model

    def _load_tokenizer(self):
        """Load the tokenizer."""
        tokenizer_path = self.checkpoint_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            # Fallback to original model
            tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["llama_model_name"])

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def predict(
        self, climate_data: torch.Tensor, text_query: str, return_attention: bool = False
    ) -> dict:
        """
        Perform inference on climate data and text query.

        Args:
            climate_data: Climate tensor [time, channels, height, width]
            text_query: Text query/question
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing predictions and optional attention weights
        """
        with torch.no_grad():
            # Prepare climate batch
            if len(climate_data.shape) == 4:
                climate_data = climate_data.unsqueeze(0)  # Add batch dimension

            climate_batch = {
                "dynamic": climate_data[:, :, :160, :, :].to(self.device),
                "static": torch.zeros(
                    climate_data.size(0), 11, climate_data.size(-2), climate_data.size(-1)
                ).to(self.device),
            }

            # Prepare text
            text_inputs = [text_query]

            # Forward pass
            outputs = self.model(climate_batch, text_inputs)

            # Extract results
            results = {
                "fused_features": outputs["fused_features"].cpu(),
                "climate_features": outputs.get("climate_features", None),
                "text_features": outputs.get("text_features", None),
                "query": text_query,
            }

            if return_attention and "attention_weights" in outputs:
                results["attention_weights"] = outputs["attention_weights"].cpu()

            return results

    def batch_predict(
        self, climate_data_list: list[torch.Tensor], text_queries: list[str], batch_size: int = 4
    ) -> list[dict]:
        """Perform batch inference."""
        results = []

        for i in range(0, len(climate_data_list), batch_size):
            batch_climate = climate_data_list[i : i + batch_size]
            batch_queries = text_queries[i : i + batch_size]

            # Stack climate data
            stacked_climate = torch.stack(batch_climate)

            # Process batch
            with torch.no_grad():
                climate_batch = {
                    "dynamic": stacked_climate[:, :, :160, :, :].to(self.device),
                    "static": torch.zeros(
                        len(batch_climate), 11, stacked_climate.size(-2), stacked_climate.size(-1)
                    ).to(self.device),
                }

                outputs = self.model(climate_batch, batch_queries)

                # Split batch results
                for j in range(len(batch_climate)):
                    results.append(
                        {
                            "fused_features": outputs["fused_features"][j : j + 1].cpu(),
                            "query": batch_queries[j],
                            "index": i + j,
                        }
                    )

        return results

    def analyze_location(
        self, climate_data: torch.Tensor, location: str, analysis_type: str = "general"
    ) -> str:
        """Generate location-specific climate analysis."""
        # Prepare query based on analysis type
        if analysis_type == "general":
            query = f"Analyze the climate conditions and patterns for {location}."
        elif analysis_type == "risks":
            query = f"What are the main climate risks and vulnerabilities for {location}?"
        elif analysis_type == "trends":
            query = f"Describe the climate trends and changes observed in {location}."
        else:
            query = f"Provide a {analysis_type} analysis of climate data for {location}."

        # Get model predictions
        results = self.predict(climate_data, query)

        # Generate text response (placeholder - you might want to add a text generation head)
        # For now, return a summary based on the fused features
        fused_features = results["fused_features"]
        feature_summary = {
            "feature_magnitude": float(torch.norm(fused_features).item()),
            "feature_mean": float(torch.mean(fused_features).item()),
            "feature_std": float(torch.std(fused_features).item()),
        }

        # Simple rule-based response (replace with actual text generation)
        magnitude = feature_summary["feature_magnitude"]
        if magnitude > 100:
            intensity = "high"
        elif magnitude > 50:
            intensity = "moderate"
        else:
            intensity = "low"

        response = (
            f"Climate analysis for {location}:\n"
            f"The climate data shows {intensity} intensity patterns with "
            f"feature magnitude of {magnitude:.2f}. "
            f"This suggests {'significant climate activity' if intensity == 'high' else 'moderate climate conditions'}."
        )

        return response


def main():
    parser = argparse.ArgumentParser(description="Multimodal climate-text inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint directory"
    )
    parser.add_argument("--climate_data", type=str, help="Path to climate data file (.pt)")
    parser.add_argument("--query", type=str, help="Text query for analysis")
    parser.add_argument("--location", type=str, help="Location name for analysis")
    parser.add_argument(
        "--analysis_type",
        type=str,
        default="general",
        choices=["general", "risks", "trends"],
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Initialize inference engine
    inference = AIFSMultimodalInference(args.checkpoint, args.device)

    # Load climate data
    if args.climate_data:
        climate_data = torch.load(args.climate_data)
        print(f"Loaded climate data: {climate_data.shape}")
    else:
        # Create dummy data for testing
        climate_data = torch.randn(2, 160, 64, 64)
        print("Using dummy climate data for testing")

    # Perform inference
    if args.query:
        # Direct query inference
        results = inference.predict(climate_data, args.query, return_attention=True)

        print(f"\nQuery: {args.query}")
        print(f"Fused features shape: {results['fused_features'].shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {torch.mean(results['fused_features']):.4f}")
        print(f"  Std: {torch.std(results['fused_features']):.4f}")
        print(f"  Norm: {torch.norm(results['fused_features']):.4f}")

    elif args.location:
        # Location-based analysis
        response = inference.analyze_location(climate_data, args.location, args.analysis_type)

        print(f"\nLocation Analysis:")
        print(response)

        # Also get raw predictions
        query = f"Analyze climate data for {args.location}"
        results = inference.predict(climate_data, query)

    else:
        print("Error: Either --query or --location must be specified")
        return

    # Save results if requested
    if args.output:
        output_data = {
            "query": args.query or f"Analysis for {args.location}",
            "climate_data_shape": list(climate_data.shape),
            "results": {
                "fused_features_shape": list(results["fused_features"].shape),
                "feature_stats": {
                    "mean": float(torch.mean(results["fused_features"])),
                    "std": float(torch.std(results["fused_features"])),
                    "norm": float(torch.norm(results["fused_features"])),
                },
            },
        }

        if args.location:
            output_data["location_analysis"] = response

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
