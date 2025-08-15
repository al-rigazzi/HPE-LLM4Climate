"""
Location-Aware Climate Text Fusion

This module extends the climate_text_fusion module with location awareness,
enabling geographic-specific climate analysis and projections.

Key Features:
- Geographic query processing
- Spatial attention masking for climate data
- Location-aware multimodal fusion
- Regional climate assessment generation
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .location_aware import GeographicResolver, SpatialCropper, LocationAwareAttention
    from .climate_text_fusion import ClimateTextFusion
except ImportError:
    from location_aware import GeographicResolver, SpatialCropper, LocationAwareAttention
    from climate_text_fusion import ClimateTextFusion

# Define fusion modes for location-aware analysis
class FusionMode:
    CROSS_ATTENTION = "cross_attention"
    CONCATENATION = "concatenate"
    ADDITIVE = "add"

class LocationAwareClimateAnalysis(nn.Module):
    """
    Location-aware climate analysis system that combines climate data,
    text queries, and geographic context for regional climate assessment.
    """

    def __init__(
        self,
        prithvi_encoder_path: str = None,
        prithvi_encoder: torch.nn.Module = None,
        llama_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        fusion_mode: str = "cross_attention",
        max_climate_tokens: int = 1024,
        max_text_length: int = 512,
        num_fusion_layers: int = 2,
        fusion_dropout: float = 0.1,
        freeze_prithvi: bool = True,
        freeze_llama: bool = True,
        device: str = "auto",
        grid_shape: Tuple[int, int] = (360, 576),
        fusion_dim: int = 512,
        num_attention_heads: int = 8
    ):
        super().__init__()

        # Store dimensions
        self.fusion_dim = fusion_dim

        # Core components
        self.geographic_resolver = GeographicResolver()
        self.spatial_cropper = SpatialCropper(grid_shape)

        # Initialize climate-text fusion with encoder path or pre-loaded encoder
        try:
            self.climate_text_fusion = ClimateTextFusion(
                prithvi_encoder_path=prithvi_encoder_path,
                prithvi_encoder=prithvi_encoder,
                llama_model_name=llama_model_name,
                fusion_mode=fusion_mode,
                max_climate_tokens=max_climate_tokens,
                max_text_length=max_text_length,
                num_fusion_layers=num_fusion_layers,
                fusion_dropout=fusion_dropout,
                freeze_prithvi=freeze_prithvi,
                freeze_llama=freeze_llama,
                device=device
            )
            # Get dimensions from the fusion model
            self.fusion_dim = self.climate_text_fusion.text_dim
        except Exception as e:
            # For testing/demo without real models or on error
            self.climate_text_fusion = None
            warnings.warn(f"Climate-text fusion initialization failed: {e}. Running in demo mode.")
            # Keep default fusion_dim

        # Location-aware attention
        self.location_attention = LocationAwareAttention(
            embed_dim=self.fusion_dim,
            num_heads=num_attention_heads
        )

        # Geographic context encoder
        self.geo_context_encoder = nn.Sequential(
            nn.Linear(4, 64),  # lat_min, lat_max, lon_min, lon_max
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.fusion_dim)
        )

        # Final fusion layers
        self.final_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )

        # Output projections
        self.risk_classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # Low, Moderate, High risk
        )

        self.trend_projector = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Trend magnitude
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Cache for resolved locations
        self._location_cache = {}

    def process_geographic_query(
        self,
        query: str
    ) -> Tuple[Optional[Dict], torch.Tensor]:
        """
        Process a geographic query to extract location and create spatial mask.

        Args:
            query: Natural language query with geographic reference

        Returns:
            Tuple of (location_info, spatial_mask)
        """
        # Check cache first
        if query in self._location_cache:
            return self._location_cache[query]

        # Extract and resolve locations
        locations = self.geographic_resolver.extract_locations(query)

        if not locations:
            # No geographic information found
            spatial_mask = torch.ones(self.spatial_cropper.n_lats, self.spatial_cropper.n_lons)
            result = (None, spatial_mask)
            self._location_cache[query] = result
            return result

        # Use the first (most confident) location
        location_text = locations[0]
        location = self.geographic_resolver.resolve_location(location_text)

        if location is None:
            spatial_mask = torch.ones(self.spatial_cropper.n_lats, self.spatial_cropper.n_lons)
            result = (None, spatial_mask)
        else:
            # Create spatial mask based on query characteristics
            mask_type = self._determine_mask_type(query)
            focus_strength = self._determine_focus_strength(query, location)

            spatial_mask = self.spatial_cropper.create_location_mask(
                location,
                mask_type=mask_type,
                focus_strength=focus_strength
            )

            location_info = {
                'name': location.name,
                'bounds': location.bounds,
                'center': location.center,
                'type': location.location_type,
                'confidence': location.confidence,
                'mask_type': mask_type,
                'focus_strength': focus_strength
            }

            result = (location_info, spatial_mask)

        self._location_cache[query] = result
        return result

    def _determine_mask_type(self, query: str) -> str:
        """Determine appropriate mask type based on query."""
        query_lower = query.lower()

        # Precise analysis needs focused masks
        if any(word in query_lower for word in ['specific', 'precise', 'exact', 'particular']):
            return "gaussian"

        # Regional analysis can use softer masks
        if any(word in query_lower for word in ['region', 'area', 'surrounding', 'nearby']):
            return "cosine"

        # Default to gaussian for most climate queries
        return "gaussian"

    def _determine_focus_strength(self, query: str, location) -> float:
        """Determine focus strength based on query and location type."""
        query_lower = query.lower()

        # High precision for coordinate queries
        if location.location_type == "coordinate":
            return 4.0

        # Strong focus for specific questions
        if any(word in query_lower for word in ['will', 'specific', 'precise', 'exactly']):
            return 3.5

        # Medium focus for countries/states
        if location.location_type in ["country", "state"]:
            return 3.0

        # Softer focus for large regions
        if location.location_type == "region":
            return 2.5

        return 3.0

    def forward(
        self,
        climate_features: torch.Tensor,
        text_query: str,
        fusion_mode: str = FusionMode.CROSS_ATTENTION
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for location-aware climate analysis.

        Args:
            climate_features: Climate data features [batch, seq_len, climate_dim]
            text_query: Natural language query with geographic context
            fusion_mode: How to fuse climate and text features

        Returns:
            Dictionary with analysis results
        """
        batch_size = climate_features.shape[0]

        # Process geographic query
        location_info, spatial_mask = self.process_geographic_query(text_query)

        # Expand spatial mask for batch
        spatial_mask = spatial_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Flatten spatial mask to match sequence dimension
        # Assuming climate_features corresponds to flattened spatial grid
        spatial_mask_flat = spatial_mask.flatten(1)  # [batch, n_lats * n_lons]

        # Adjust to match sequence length of climate features
        seq_len = climate_features.shape[1]
        if spatial_mask_flat.shape[1] != seq_len:
            # Interpolate or crop spatial mask to match sequence length
            spatial_mask_flat = F.interpolate(
                spatial_mask_flat.unsqueeze(1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(1)

        # Standard climate-text fusion
        if self.climate_text_fusion:
            # Since we have pre-extracted features, use the components directly
            text_features, text_attention_mask = self.climate_text_fusion.encode_text([text_query] * batch_size)

            # Project climate features to text space
            climate_projected = self.climate_text_fusion.climate_projector(climate_features)

            # Apply fusion (simplified for pre-extracted features)
            if hasattr(self.climate_text_fusion, 'fusion_layers') and len(self.climate_text_fusion.fusion_layers) > 0:
                fused_features = self.climate_text_fusion.fusion_layers[0](
                    text_features, climate_projected, climate_projected
                )[0]  # Cross-attention output
            else:
                # Simple additive fusion as fallback
                min_len = min(text_features.shape[1], climate_projected.shape[1])
                fused_features = text_features[:, :min_len] + climate_projected[:, :min_len]

            # Apply output projection if available
            if hasattr(self.climate_text_fusion, 'output_projection'):
                fused_features = self.climate_text_fusion.output_projection(fused_features)
        else:
            # Demo mode - create mock fusion features
            text_attention_mask = None
            # Project climate features to fusion dimension
            if not hasattr(self, 'demo_projector'):
                self.demo_projector = nn.Linear(climate_features.shape[-1], self.fusion_dim)
            fused_features = self.demo_projector(climate_features)        # Apply location-aware attention
        # Apply location-aware attention
        # Ensure fused_features has the right shape for attention [B, N, C]
        if len(fused_features.shape) == 2:
            fused_features = fused_features.unsqueeze(0)  # Add batch dimension

        # Handle spatial mask mismatch for text-based features
        if spatial_mask_flat is not None and fused_features.shape[1] != spatial_mask_flat.shape[-1]:
            # For text-based features, create a simple uniform mask
            spatial_mask_flat = torch.ones(batch_size, 1, 1, fused_features.shape[1], device=fused_features.device)

        location_attended = self.location_attention(
            fused_features,
            spatial_mask=spatial_mask_flat
        )

        # Create geographic context encoding if location found
        if location_info is not None:
            bounds = location_info['bounds']
            geo_context = torch.tensor([
                bounds['lat_min'], bounds['lat_max'],
                bounds['lon_min'], bounds['lon_max']
            ], dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)

            if next(self.parameters()).is_cuda:
                geo_context = geo_context.cuda()

            geo_encoding = self.geo_context_encoder(geo_context)  # [batch, fusion_dim]
        else:
            # No geographic context - use zeros
            geo_encoding = torch.zeros(
                batch_size, fused_features.shape[-1],
                device=fused_features.device, dtype=fused_features.dtype
            )

        # Pool location-attended features
        pooled_features = location_attended.mean(dim=1)  # [batch, fusion_dim]

        # Final fusion with geographic context
        combined = torch.cat([pooled_features, geo_encoding], dim=-1)
        final_features = self.final_fusion(combined)

        # Generate predictions
        climate_risk = self.risk_classifier(final_features)
        trend_magnitude = self.trend_projector(final_features)
        confidence = self.confidence_estimator(final_features)

        # Spatial attention weights for visualization
        attention_weights = spatial_mask_flat.mean(dim=0)  # Average across batch

        return {
            'fused_features': final_features,
            'climate_risk': climate_risk,
            'trend_magnitude': trend_magnitude,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'location_info': location_info,
            'spatial_mask': spatial_mask,
            # Include text attention mask if available
            **({"text_attention_mask": text_attention_mask} if self.climate_text_fusion and 'text_attention_mask' in locals() else {})
        }

    def analyze_location_query(
        self,
        climate_features: torch.Tensor,
        query: str,
        return_visualization: bool = False
    ) -> Dict[str, Union[str, float, torch.Tensor]]:
        """
        High-level interface for location-aware climate analysis.

        Args:
            climate_features: Climate data features
            query: Natural language query
            return_visualization: Whether to return visualization data

        Returns:
            Dictionary with analysis results and interpretations
        """
        # Forward pass
        result = self.forward(climate_features, query)

        # Interpret results
        risk_probs = F.softmax(result['climate_risk'], dim=-1)
        risk_categories = ['Low Risk', 'Moderate Risk', 'High Risk']
        risk_prediction = risk_categories[risk_probs.argmax(dim=-1).item()]
        risk_confidence = risk_probs.max().item()

        trend_value = result['trend_magnitude'].item()
        overall_confidence = result['confidence'].item()

        # Generate interpretation
        location_info = result['location_info']
        if location_info:
            location_name = location_info['name']
            location_type = location_info['type']
        else:
            location_name = "Global"
            location_type = "global"

        interpretation = self._generate_interpretation(
            query, location_name, location_type, risk_prediction,
            trend_value, risk_confidence, overall_confidence
        )

        analysis_result = {
            'query': query,
            'location': location_name,
            'location_type': location_type,
            'climate_risk': risk_prediction,
            'risk_confidence': risk_confidence,
            'trend_magnitude': trend_value,
            'overall_confidence': overall_confidence,
            'interpretation': interpretation
        }

        if return_visualization:
            analysis_result.update({
                'attention_weights': result['attention_weights'],
                'spatial_mask': result['spatial_mask'],
                'location_bounds': location_info['bounds'] if location_info else None
            })

        return analysis_result

    def _generate_interpretation(
        self,
        query: str,
        location: str,
        location_type: str,
        risk: str,
        trend: float,
        risk_conf: float,
        overall_conf: float
    ) -> str:
        """Generate human-readable interpretation of results."""

        # Trend interpretation
        if abs(trend) < 0.3:
            trend_desc = "stable conditions"
        elif trend > 0:
            if trend > 0.7:
                trend_desc = "significant worsening trends"
            else:
                trend_desc = "concerning trends"
        else:
            if trend < -0.7:
                trend_desc = "significant improvement trends"
            else:
                trend_desc = "improving trends"

        # Confidence interpretation
        if overall_conf > 0.8:
            conf_desc = "high confidence"
        elif overall_conf > 0.6:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "limited confidence"

        # Location-specific context
        if location_type == "coordinate":
            loc_context = f"the specific coordinates {location}"
        elif location_type == "country":
            loc_context = f"the country of {location}"
        elif location_type == "state":
            loc_context = f"the state of {location}"
        elif location_type == "region":
            loc_context = f"the {location} region"
        else:
            loc_context = "the global scale"

        interpretation = f"""
Based on the climate analysis for {loc_context}, the assessment indicates {risk.lower()}
with {trend_desc}. The model has {conf_desc} in this projection
(confidence: {overall_conf:.1%}).

Risk assessment confidence is {risk_conf:.1%}, suggesting the model is
{'quite certain' if risk_conf > 0.8 else 'moderately confident' if risk_conf > 0.6 else 'uncertain'}
about the risk classification.
        """.strip()

        return interpretation

def demo_location_aware_analysis():
    """Demonstrate location-aware climate analysis."""
    print("🌍 Location-Aware Climate Analysis Demo\n")

    # Initialize model
    model = LocationAwareClimateAnalysis()
    model.eval()

    # Simulate climate features (e.g., from Prithvi encoder)
    batch_size = 1
    seq_len = 1024  # Number of spatial patches or time series points
    climate_dim = 768

    climate_features = torch.randn(batch_size, seq_len, climate_dim)

    # Test queries with different geographic contexts
    test_queries = [
        "What crops will be viable in Sweden by 2050?",
        "How will drought patterns change in Arizona?",
        "Climate resilience in the Mediterranean region",
        "Tornado frequency changes near 40.7°N, 74.0°W",
        "Arctic ice melting projections"
    ]

    with torch.no_grad():
        for i, query in enumerate(test_queries):
            print(f"Query {i+1}: {query}")
            print("-" * 50)

            # Analyze query
            result = model.analyze_location_query(
                climate_features, query, return_visualization=True
            )

            print(f"Location: {result['location']} ({result['location_type']})")
            print(f"Climate Risk: {result['climate_risk']}")
            print(f"Risk Confidence: {result['risk_confidence']:.1%}")
            print(f"Trend Magnitude: {result['trend_magnitude']:.2f}")
            print(f"Overall Confidence: {result['overall_confidence']:.1%}")
            print(f"\nInterpretation:")
            print(result['interpretation'])

            if result.get('location_bounds'):
                bounds = result['location_bounds']
                print(f"\nGeographic Bounds:")
                print(f"  Latitude: {bounds['lat_min']:.1f}° to {bounds['lat_max']:.1f}°")
                print(f"  Longitude: {bounds['lon_min']:.1f}° to {bounds['lon_max']:.1f}°")

            print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    demo_location_aware_analysis()
