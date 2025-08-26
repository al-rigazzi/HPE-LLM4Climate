"""
AIFS Location-Aware Components

This module provides location-aware components specifically designed for AIFS,
including geographic resolvers, spatial croppers, and location-aware attention
mechanisms for spatially-informed climate analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.aifs_encoder_utils import AIFSEncoderWrapper
from ..utils.location_utils import GridUtils, LocationUtils, SpatialEncoder


class AIFSGeographicResolver(nn.Module):
    """
    Geographic resolver for AIFS-based climate analysis.

    This module resolves geographic locations and provides spatial context
    for climate data analysis using AIFS encodings.
    """

    def __init__(
        self,
        coordinate_dim: int = 64,
        context_radius_km: float = 500.0,
        grid_resolution: float = 0.25,
        device: str = "cpu",
    ):
        """
        Initialize geographic resolver.

        Args:
            coordinate_dim: Dimension for coordinate encodings
            context_radius_km: Radius for spatial context in kilometers
            grid_resolution: Grid resolution in degrees
            device: Device to run on
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.context_radius_km = context_radius_km
        self.grid_resolution = grid_resolution
        self.device = device

        # Initialize spatial encoder
        self.spatial_encoder = SpatialEncoder(encoding_dim=coordinate_dim)

        # Grid utilities for spatial operations
        self.grid_utils = GridUtils(
            lat_range=(-90, 90), lon_range=(-180, 180), resolution=grid_resolution
        )

        # Coordinate embedding network
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(coordinate_dim, coordinate_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(coordinate_dim * 2, coordinate_dim),
            nn.LayerNorm(coordinate_dim),
        )

        # Context aggregation network
        self.context_aggregator = nn.Sequential(
            nn.Linear(coordinate_dim * 2, coordinate_dim),
            nn.ReLU(),
            nn.Linear(coordinate_dim, coordinate_dim // 2),
            nn.ReLU(),
            nn.Linear(coordinate_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode_location(self, lat: float, lon: float) -> torch.Tensor:
        """
        Encode a geographic location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Location encoding tensor
        """
        # Validate and normalize coordinates
        lat, lon = LocationUtils.normalize_coordinates(lat, lon)

        # Create spatial encoding
        encoding = self.spatial_encoder.encode_coordinates(lat, lon)
        encoding = encoding.to(self.device)

        # Apply embedding network
        embedded = self.coordinate_embedding(encoding.unsqueeze(0))

        return torch.as_tensor(embedded.squeeze(0))

    def encode_location_batch(self, coordinates: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Encode a batch of geographic locations.

        Args:
            coordinates: List of (lat, lon) tuples

        Returns:
            Batch of location encodings
        """
        encodings = []

        for lat, lon in coordinates:
            encoding = self.encode_location(lat, lon)
            encodings.append(encoding)

        return torch.stack(encodings)

    def get_spatial_context(
        self, center_lat: float, center_lon: float, nearby_locations: List[Tuple[float, float]]
    ) -> torch.Tensor:
        """
        Get spatial context around a center location.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            nearby_locations: List of nearby (lat, lon) coordinates

        Returns:
            Spatial context encoding
        """
        # Encode center location
        center_encoding = self.encode_location(center_lat, center_lon)

        if not nearby_locations:
            return center_encoding

        # Encode nearby locations
        nearby_encodings = []
        weights = []

        for lat, lon in nearby_locations:
            # Calculate distance
            distance = LocationUtils.haversine_distance(center_lat, center_lon, lat, lon)

            if distance <= self.context_radius_km:
                # Encode location
                encoding = self.encode_location(lat, lon)
                nearby_encodings.append(encoding)

                # Calculate distance-based weight
                weight = np.exp(-distance / (self.context_radius_km / 3))
                weights.append(weight)

        if not nearby_encodings:
            return center_encoding

        # Stack and weight nearby encodings
        nearby_stack = torch.stack(nearby_encodings)
        weight_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.sum()  # Normalize

        # Weighted aggregation
        weighted_context = (nearby_stack * weight_tensor.unsqueeze(1)).sum(dim=0)

        # Combine center and context
        combined = torch.cat([center_encoding, weighted_context])
        context_weight = self.context_aggregator(combined)

        # Blend center and context
        final_context = (1 - context_weight) * center_encoding + context_weight * weighted_context

        return torch.as_tensor(final_context)

    def resolve_text_location(self, text: str) -> Tuple[float, float] | None:
        """
        Extract location from text description.

        Args:
            text: Text containing location information

        Returns:
            Extracted (lat, lon) or None if not found
        """
        # Import text processor locally to avoid circular imports
        from ..utils.text_utils import ClimateTextProcessor

        processor = ClimateTextProcessor()
        locations = processor.extract_locations(text)

        for location in locations:
            if location["type"] == "coordinates":
                return location["latitude"], location["longitude"]

        # TODO: Add geocoding for city names, regions, etc.
        return None


class AIFSSpatialCropper(nn.Module):
    """
    Spatial cropper for AIFS climate data.

    Extracts relevant spatial regions from global climate data based on
    location queries and spatial attention mechanisms.
    """

    def __init__(
        self,
        grid_resolution: float = 0.25,
        crop_size_km: float = 1000.0,
        attention_heads: int = 8,
        feature_dim: int = 1024,
        device: str = "cpu",
    ):
        """
        Initialize spatial cropper.

        Args:
            grid_resolution: Grid resolution in degrees
            crop_size_km: Size of cropped region in kilometers
            attention_heads: Number of attention heads
            feature_dim: Feature dimension for attention
            device: Device to run on
        """
        super().__init__()

        self.grid_resolution = grid_resolution
        self.crop_size_km = crop_size_km
        self.feature_dim = feature_dim
        self.device = device

        # Grid utilities
        self.grid_utils = GridUtils(
            lat_range=(-90, 90), lon_range=(-180, 180), resolution=grid_resolution
        )

        # Spatial attention for region selection
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=attention_heads, dropout=0.1, batch_first=True
        )

        # Location query projection
        self.location_query = nn.Linear(64, feature_dim)  # 64 is spatial encoding dim

        # Feature projection for climate data
        self.climate_projection = nn.Linear(1, feature_dim)  # Project scalar values

        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

    def create_spatial_features(
        self, climate_data: torch.Tensor, coordinates: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Create spatial features for climate data.

        Args:
            climate_data: Climate data tensor (variables, lat, lon)
            coordinates: Grid coordinates (lat_size, lon_size)

        Returns:
            Spatial features tensor
        """
        lat_size, lon_size = coordinates

        # Flatten spatial dimensions
        if climate_data.dim() == 3:  # (vars, lat, lon)
            reshaped = climate_data.permute(1, 2, 0)  # (lat, lon, vars)
            flattened = reshaped.reshape(-1, climate_data.size(0))  # (lat*lon, vars)
        else:  # (batch, vars, lat, lon)
            batch_size = climate_data.size(0)
            reshaped = climate_data.permute(0, 2, 3, 1)  # (batch, lat, lon, vars)
            flattened = reshaped.reshape(
                batch_size, -1, climate_data.size(1)
            )  # (batch, lat*lon, vars)

        # Project to feature space (sum over variable dimension for now)
        if flattened.dim() == 2:
            features = self.climate_projection(flattened.sum(dim=1, keepdim=True))
        else:
            features = self.climate_projection(flattened.sum(dim=2, keepdim=True))

        return torch.as_tensor(features)

    def crop_region(
        self, climate_data: torch.Tensor, center_lat: float, center_lon: float
    ) -> torch.Tensor:
        """
        Crop a region around a center point.

        Args:
            climate_data: Global climate data
            center_lat: Center latitude
            center_lon: Center longitude

        Returns:
            Cropped region data
        """
        # Extract region using grid utilities
        cropped = self.grid_utils.extract_region(
            climate_data, center_lat, center_lon, self.crop_size_km
        )

        return torch.as_tensor(cropped)

    def attention_crop(
        self, climate_data: torch.Tensor, query_location: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crop region using attention mechanism.

        Args:
            climate_data: Climate data tensor
            query_location: Location query encoding

        Returns:
            Attention-weighted climate features
        """
        # Create spatial features
        coordinates = climate_data.shape[-2:]  # (lat, lon)
        spatial_features = self.create_spatial_features(
            climate_data, (int(coordinates[0]), int(coordinates[1]))
        )

        # Project location query
        query = self.location_query(query_location.unsqueeze(0))  # (1, feature_dim)

        # Apply attention
        attended_features, attention_weights = self.spatial_attention(
            query, spatial_features, spatial_features
        )

        # Normalize and project
        output = self.norm(attended_features)
        output = self.output_projection(output)

        return output.squeeze(0), attention_weights

    def forward(
        self,
        climate_data: torch.Tensor,
        center_lat: float,
        center_lon: float,
        use_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of spatial cropper.

        Args:
            climate_data: Global climate data
            center_lat: Center latitude
            center_lon: Center longitude
            use_attention: Whether to use attention-based cropping

        Returns:
            Dictionary containing cropping results
        """
        results = {}

        # Traditional spatial cropping
        cropped_region = self.crop_region(climate_data, center_lat, center_lon)
        results["cropped_region"] = cropped_region

        if use_attention:
            # Create location encoding
            spatial_encoder = SpatialEncoder(encoding_dim=64)
            location_encoding = spatial_encoder.encode_coordinates(center_lat, center_lon)
            location_encoding = location_encoding.to(self.device)

            # Attention-based cropping
            attended_features, attention_weights = self.attention_crop(
                climate_data, location_encoding
            )

            results["attended_features"] = attended_features
            results["attention_weights"] = attention_weights

        return results


class AIFSLocationAwareAttention(nn.Module):
    """
    Location-aware attention mechanism for AIFS climate analysis.

    This module implements attention that is conditioned on geographic location,
    allowing for spatially-informed climate analysis.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        location_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize location-aware attention.

        Args:
            feature_dim: Dimension of input features
            location_dim: Dimension of location encodings
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for attention softmax
            device: Device to run on
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.location_dim = location_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.temperature = temperature
        self.device = device

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Location projection
        self.location_projection = nn.Linear(location_dim, feature_dim)

        # Attention projections
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)

        # Location-aware bias
        self.location_bias = nn.Linear(feature_dim, num_heads)

        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, locations: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of location-aware attention.

        Args:
            features: Input features (batch, seq_len, feature_dim)
            locations: Location encodings (batch, seq_len, location_dim)
            mask: Attention mask (optional)

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len = features.shape[:2]

        # Project locations to feature space
        location_features = self.location_projection(locations)

        # Combine features with location information
        combined_features = features + location_features

        # Create query, key, value
        queries = self.query_projection(combined_features)
        keys = self.key_projection(combined_features)
        values = self.value_projection(combined_features)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim**0.5)

        # Add location-aware bias
        location_bias = self.location_bias(location_features)  # (batch, seq_len, num_heads)
        location_bias = location_bias.unsqueeze(2)  # (batch, seq_len, 1, num_heads)
        location_bias = location_bias.transpose(2, 3)  # (batch, seq_len, num_heads, 1)
        location_bias = location_bias.transpose(1, 2)  # (batch, num_heads, seq_len, 1)

        attention_scores = attention_scores + location_bias

        # Apply temperature scaling
        attention_scores = attention_scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Reshape back
        attended_values = (
            attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        )

        # Apply output projection
        output = self.output_projection(attended_values)

        # Residual connection and normalization
        output = self.norm(output + features)

        # Average attention weights across heads for visualization
        avg_attention_weights = attention_weights.mean(dim=1)

        return output, avg_attention_weights


def test_aifs_location_aware():
    """Test AIFS location-aware components."""
    print("🗺️ Testing AIFS Location-Aware Components")
    print("=" * 45)

    # Test geographic resolver
    print("Testing Geographic Resolver...")
    resolver = AIFSGeographicResolver(coordinate_dim=64, device="cpu")

    # Test single location encoding
    london_lat, london_lon = 51.5074, -0.1278
    location_encoding = resolver.encode_location(london_lat, london_lon)
    print(f"Location encoding shape: {location_encoding.shape}")

    # Test batch encoding
    coordinates = [(51.5074, -0.1278), (48.8566, 2.3522), (40.7128, -74.0060)]
    batch_encodings = resolver.encode_location_batch(coordinates)
    print(f"Batch encodings shape: {batch_encodings.shape}")

    # Test spatial context
    nearby_coords = [(51.4, -0.2), (51.6, -0.1), (51.5, 0.0)]
    context = resolver.get_spatial_context(london_lat, london_lon, nearby_coords)
    print(f"Spatial context shape: {context.shape}")

    # Test spatial cropper
    print("\\nTesting Spatial Cropper...")
    cropper = AIFSSpatialCropper(grid_resolution=1.0, crop_size_km=500, device="cpu")

    # Create synthetic global climate data
    global_data = torch.randn(5, 181, 361)  # (vars, lat, lon)

    crop_results = cropper(global_data, london_lat, london_lon, use_attention=True)
    print(f"Cropped region shape: {crop_results['cropped_region'].shape}")
    if "attended_features" in crop_results:
        print(f"Attended features shape: {crop_results['attended_features'].shape}")

    # Test location-aware attention
    print("\\nTesting Location-Aware Attention...")
    attention = AIFSLocationAwareAttention(
        feature_dim=256, location_dim=64, num_heads=8, device="cpu"
    )

    # Create test data
    batch_size, seq_len = 4, 10
    features = torch.randn(batch_size, seq_len, 256)
    locations = torch.randn(batch_size, seq_len, 64)

    attended_features, attention_weights = attention(features, locations)
    print(f"Attended features shape: {attended_features.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    print("✅ All location-aware tests passed!")


if __name__ == "__main__":
    test_aifs_location_aware()
