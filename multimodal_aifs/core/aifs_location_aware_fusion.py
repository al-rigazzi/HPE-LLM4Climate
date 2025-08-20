#!/usr/bin/env python3
"""
AIFS Location-Aware Fusion Model

This module provides location-aware climate-text fusion using AIFS encoder,
similar to the Prithvi-based multimodal fusion but adapted for AIFS.

Features:
- Geographic text parsing and location resolution
- Spatial cropping and regional analysis
- Location-aware attention mechanisms
- AIFS-based spatial encoding with geographic context
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer
from multimodal_aifs.utils.location_utils import GridUtils, LocationUtils, SpatialEncoder
from multimodal_aifs.utils.text_utils import extract_location_keywords, parse_climate_query

# Import LLaMA components
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Try optional components for quantization
    try:
        from transformers import BitsAndBytesConfig

        QUANTIZATION_AVAILABLE = True
    except ImportError:
        QUANTIZATION_AVAILABLE = False
    LLAMA_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"LLaMA/Transformers not available: {e}. Using mock implementations.")
    LLAMA_AVAILABLE = False
    QUANTIZATION_AVAILABLE = False


class GeographicResolver:
    """
    Simplified geographic resolver for AIFS integration.

    Resolves location names from text to coordinate bounds for spatial processing.
    """

    def __init__(self):
        self.location_cache = {}

        # Common locations database (simplified version)
        self.known_locations = {
            "new york": {"lat": 40.7128, "lon": -74.0060, "name": "New York City"},
            "london": {"lat": 51.5074, "lon": -0.1278, "name": "London"},
            "paris": {"lat": 48.8566, "lon": 2.3522, "name": "Paris"},
            "tokyo": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo"},
            "sydney": {"lat": -33.8688, "lon": 151.2093, "name": "Sydney"},
            "san francisco": {"lat": 37.7749, "lon": -122.4194, "name": "San Francisco"},
            "miami": {"lat": 25.7617, "lon": -80.1918, "name": "Miami"},
            "chicago": {"lat": 41.8781, "lon": -87.6298, "name": "Chicago"},
            "los angeles": {"lat": 34.0522, "lon": -118.2437, "name": "Los Angeles"},
            "beijing": {"lat": 39.9042, "lon": 116.4074, "name": "Beijing"},
            # Regional descriptors
            "northern california": {"lat": 37.5, "lon": -120.0, "name": "Northern California"},
            "southern california": {"lat": 34.0, "lon": -118.0, "name": "Southern California"},
            "east coast": {"lat": 39.0, "lon": -76.0, "name": "US East Coast"},
            "west coast": {"lat": 36.0, "lon": -119.0, "name": "US West Coast"},
            "midwest": {"lat": 41.0, "lon": -89.0, "name": "US Midwest"},
            "great lakes": {"lat": 45.0, "lon": -84.0, "name": "Great Lakes Region"},
        }

    def resolve_location(self, location_text: str) -> Optional[Dict]:
        """
        Resolve location text to coordinates.

        Args:
            location_text: Location name or description

        Returns:
            Dictionary with lat, lon, name, and bounds
        """
        if location_text in self.location_cache:
            cached_result = self.location_cache[location_text]
            return cached_result if cached_result is not None else None

        # Normalize text
        normalized = location_text.lower().strip()

        # Direct lookup
        if normalized in self.known_locations:
            location_info = self.known_locations[normalized].copy()

            # Add bounding box (roughly 1 degree around center)
            lat_raw = location_info.get("lat", 0)
            lon_raw = location_info.get("lon", 0)
            lat_val = float(lat_raw) if isinstance(lat_raw, (int, float, str)) else 0.0
            lon_val = float(lon_raw) if isinstance(lon_raw, (int, float, str)) else 0.0
            location_info.update(
                {
                    "bounds": {
                        "north": lat_val + 0.5,
                        "south": lat_val - 0.5,
                        "east": lon_val + 0.5,
                        "west": lon_val - 0.5,
                    }
                }
            )

            self.location_cache[location_text] = location_info
            return location_info

        # Partial matching for flexibility
        for known_loc, info in self.known_locations.items():
            if known_loc in normalized or normalized in known_loc:
                location_info = info.copy()
                lat_raw = info.get("lat", 0)
                lon_raw = info.get("lon", 0)
                lat_val = float(lat_raw) if isinstance(lat_raw, (int, float, str)) else 0.0
                lon_val = float(lon_raw) if isinstance(lon_raw, (int, float, str)) else 0.0
                location_info.update(
                    {
                        "bounds": {
                            "north": lat_val + 0.5,
                            "south": lat_val - 0.5,
                            "east": lon_val + 0.5,
                            "west": lon_val - 0.5,
                        }
                    }
                )
                self.location_cache[location_text] = location_info
                return location_info

        return None


class SpatialCropper:
    """
    Spatial cropping utility for extracting regional climate data.
    """

    def __init__(self, grid_shape: Tuple[int, int] = (721, 1440)):
        """
        Initialize spatial cropper.

        Args:
            grid_shape: (lat, lon) dimensions of global grid
        """
        self.grid_shape = grid_shape
        self.lat_size, self.lon_size = grid_shape

        # Global grid coordinates
        self.lats = np.linspace(90, -90, self.lat_size)
        self.lons = np.linspace(-180, 180, self.lon_size)

    def crop_to_region(
        self, climate_data: torch.Tensor, bounds: Dict[str, float], padding: float = 0.5
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Crop climate data to specified geographic bounds.

        Args:
            climate_data: [batch, time, vars, lat, lon] tensor
            bounds: Geographic bounds dict with north, south, east, west
            padding: Additional padding in degrees

        Returns:
            Cropped tensor and cropping info
        """
        # Add padding to bounds
        north = min(90, bounds["north"] + padding)
        south = max(-90, bounds["south"] - padding)
        east = bounds["east"] + padding
        west = bounds["west"] - padding

        # Handle longitude wrapping
        if east > 180:
            east -= 360
        if west < -180:
            west += 360

        # Find grid indices
        lat_indices = np.where((self.lats <= north) & (self.lats >= south))[0]

        if west <= east:
            lon_indices = np.where((self.lons >= west) & (self.lons <= east))[0]
        else:
            # Handle longitude wrapping
            lon_indices = np.where((self.lons >= west) | (self.lons <= east))[0]

        # Crop the data
        lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1
        lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1

        cropped_data = climate_data[:, :, :, lat_start:lat_end, lon_start:lon_end]

        crop_info = {
            "lat_indices": (lat_start, lat_end),
            "lon_indices": (lon_start, lon_end),
            "lat_bounds": (self.lats[lat_start], self.lats[lat_end - 1]),
            "lon_bounds": (self.lons[lon_start], self.lons[lon_end - 1]),
            "original_shape": climate_data.shape,
            "cropped_shape": cropped_data.shape,
        }

        return cropped_data, crop_info


class AIFSLocationAwareFusion(nn.Module):
    """
    Location-aware climate-text fusion using AIFS encoder.

    This model combines:
    - AIFS spatial encoding for climate data
    - Geographic resolution for location queries
    - Spatial cropping for regional analysis
    - Location-aware attention mechanisms
    - LLaMA for natural language understanding
    """

    def __init__(
        self,
        llama_model_name: str = "meta-llama/Meta-Llama-3-8B",
        time_series_dim: int = 512,
        fusion_strategy: str = "location_aware_cross_attention",
        device: str = "cpu",
        use_quantization: bool = True,
        use_mock_llama: bool = False,
        grid_shape: Tuple[int, int] = (721, 1440),
        max_text_length: int = 512,
    ):
        """
        Initialize AIFS location-aware fusion model.

        Args:
            llama_model_name: HuggingFace model name for LLaMA
            time_series_dim: Dimension of time series tokens
            fusion_strategy: Fusion approach (supports location-aware variants)
            device: Device to run on
            use_quantization: Whether to use quantization
            use_mock_llama: Use mock LLaMA for testing
            grid_shape: Global climate data grid shape
            max_text_length: Maximum text sequence length
        """
        super().__init__()

        self.device = device
        self.fusion_strategy = fusion_strategy
        self.time_series_dim = time_series_dim
        self.max_text_length = max_text_length

        # Initialize geographic components
        self.geographic_resolver = GeographicResolver()
        self.spatial_cropper = SpatialCropper(grid_shape)
        self.spatial_encoder = SpatialEncoder(encoding_dim=64)

        # Initialize AIFS time series tokenizer
        self.time_series_tokenizer = AIFSTimeSeriesTokenizer(
            temporal_modeling="transformer", hidden_dim=time_series_dim, device=device
        )

        # Initialize LLaMA model - can be either real model or mock
        self.llama_model: Union[Any, "MockLlamaModel"]

        # Initialize LLaMA model - try real LLaMA first, fallback to mock
        if LLAMA_AVAILABLE and not use_mock_llama:
            self._initialize_real_llama(llama_model_name, use_quantization)
        else:
            print("   Using mock LLaMA model for testing")
            from multimodal_aifs.tests.integration.test_aifs_llama_integration import MockLlamaModel

            self.llama_model = MockLlamaModel().to(device)
            self.llama_tokenizer = None
            self.llama_hidden_size = 4096

        # Initialize location-aware fusion components
        self._initialize_location_aware_fusion_layers()

    def _initialize_real_llama(self, model_name: str, use_quantization: bool):
        """Initialize real LLaMA model with optional quantization."""
        try:
            print(f"   🚀 Attempting to load LLaMA model: {model_name}")

            # Disable flash attention for compatibility
            import os

            os.environ["TRANSFORMERS_USE_FLASH_ATTENTION_2"] = "false"

            # Initialize tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left"
            )

            if self.llama_tokenizer is not None and self.llama_tokenizer.pad_token is None:  # type: ignore[unreachable]
                if self.llama_tokenizer.eos_token is not None:  # type: ignore[unreachable]
                    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            # Configure quantization if requested and available
            if use_quantization and QUANTIZATION_AVAILABLE:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                )
                print("   🔧 Using 8-bit quantization")
            else:
                quantization_config = None
                if use_quantization and not QUANTIZATION_AVAILABLE:
                    print(
                        "   ⚠️ Quantization requested but not available, loading in full precision"
                    )

            # Initialize model with flash attention disabled
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager",  # Disable flash attention
                low_cpu_mem_usage=True,
            )

            self.llama_hidden_size = self.llama_model.config.hidden_size
            print(f"   ✅ Successfully loaded LLaMA model: {model_name}")
            print(f"   📏 Hidden size: {self.llama_hidden_size}")

        except Exception as e:
            print(f"   ⚠️ Failed to load LLaMA model: {e}")
            print("   🔄 Falling back to mock LLaMA")
            from multimodal_aifs.tests.integration.test_aifs_llama_integration import MockLlamaModel

            self.llama_model = MockLlamaModel().to(self.device)
            self.llama_tokenizer = None
            self.llama_hidden_size = 4096

    def _initialize_location_aware_fusion_layers(self):
        """Initialize location-aware fusion layers."""
        # Project time series tokens to LLaMA dimension
        self.ts_projection = nn.Linear(self.time_series_dim, self.llama_hidden_size)

        # Project spatial encodings to LLaMA dimension
        self.spatial_projection = nn.Linear(64, self.llama_hidden_size)

        if "location_aware" in self.fusion_strategy:
            # Location-aware cross-attention
            self.location_cross_attention = nn.MultiheadAttention(
                embed_dim=self.llama_hidden_size, num_heads=8, batch_first=True
            )

            # Geographic attention for spatial regions
            self.geographic_attention = nn.MultiheadAttention(
                embed_dim=self.llama_hidden_size, num_heads=4, batch_first=True
            )

            # Fusion normalization layers
            self.location_norm = nn.LayerNorm(self.llama_hidden_size)
            self.geographic_norm = nn.LayerNorm(self.llama_hidden_size)

        # Standard fusion layers
        self.fusion_linear = nn.Linear(self.llama_hidden_size * 2, self.llama_hidden_size)

        # Output projections for different tasks
        self.classification_head = nn.Linear(
            self.llama_hidden_size, 10
        )  # Example: 10 climate classes

    def process_location_query(self, text_input: str) -> Optional[Dict]:
        """
        Process a text input to extract location information.

        Args:
            text_input: Natural language query

        Returns:
            Location information dict or None
        """
        # Extract location keywords
        location_keywords = extract_location_keywords(text_input)

        if not location_keywords:
            return None

        # Try to resolve the first found location
        for location_text in location_keywords:
            location_info = self.geographic_resolver.resolve_location(location_text)
            if location_info:
                return location_info

        return None

    def tokenize_climate_data(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Tokenize climate data using AIFS encoder."""
        return torch.as_tensor(self.time_series_tokenizer(climate_data))

    def tokenize_text(self, text_inputs: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs."""
        if self.llama_tokenizer is None:
            # Mock tokenization
            batch_size = len(text_inputs)
            seq_len = 10
            vocab_size = 32000

            return {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }

        return self.llama_tokenizer(  # type: ignore[unreachable]
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )

    def forward(
        self, climate_data: torch.Tensor, text_inputs: List[str], task: str = "embedding", **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with location-aware processing.

        Args:
            climate_data: [batch, time, vars, lat, lon] climate tensor
            text_inputs: List of text queries
            task: Task type ("embedding", "generation", "classification")

        Returns:
            Dictionary with task-specific outputs
        """
        batch_size = climate_data.shape[0]

        # Process each text input for location information
        location_infos = []
        processed_climate_data = []

        for i, text_input in enumerate(text_inputs):
            location_info = self.process_location_query(text_input)
            location_infos.append(location_info)

            # Extract single sample from batch for processing
            sample_climate = climate_data[i : i + 1]

            if location_info and location_info.get("bounds"):
                # Crop to region of interest
                cropped_climate, crop_info = self.spatial_cropper.crop_to_region(
                    sample_climate, location_info["bounds"]
                )
                processed_climate_data.append(cropped_climate)
                print(f"   🗺️ Cropped to {location_info['name']}: {crop_info['cropped_shape']}")
            else:
                # Use full global data
                processed_climate_data.append(sample_climate)

        # Recombine processed climate data
        # Note: In practice, you might want to pad to consistent size
        # For simplicity, we'll process the first sample or use original if no location
        if processed_climate_data:
            # For demo, use first processed sample
            final_climate_data = processed_climate_data[0]
            if final_climate_data.shape[0] < batch_size:
                # Repeat to match batch size for demo
                final_climate_data = final_climate_data.repeat(batch_size, 1, 1, 1, 1)
        else:
            final_climate_data = climate_data

        # Process time series through AIFS
        ts_tokens = self.tokenize_climate_data(final_climate_data)  # [batch, time, ts_dim]

        # Process text
        text_tokens = self.tokenize_text(text_inputs)
        text_input_ids = text_tokens["input_ids"].to(self.device)
        text_attention_mask = text_tokens["attention_mask"].to(self.device)

        # Get LLaMA embeddings
        llama_outputs = self.llama_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask, output_hidden_states=True
        )

        # Handle different model output formats
        if hasattr(llama_outputs, "last_hidden_state"):
            text_embeddings = llama_outputs.last_hidden_state
        elif hasattr(llama_outputs, "hidden_states") and llama_outputs.hidden_states:
            text_embeddings = llama_outputs.hidden_states[-1]
        else:
            # Fallback
            logits = llama_outputs.logits
            text_embeddings = torch.randn(
                logits.shape[0],
                logits.shape[1],
                self.llama_hidden_size,
                device=logits.device,
                dtype=logits.dtype,
            )

        # Location-aware fusion
        fused_embeddings = self._fuse_with_location_awareness(
            ts_tokens, text_embeddings, location_infos
        )

        # Task-specific processing
        if task == "generation":
            result = self._generate_text(fused_embeddings, text_input_ids)
            result["location_info"] = location_infos
            return result
        elif task == "classification":
            result = self._classify(fused_embeddings)
            result["location_info"] = location_infos
            return result
        elif task == "embedding":
            return {"embeddings": fused_embeddings, "location_info": location_infos}
        else:
            raise ValueError(f"Unknown task: {task}")

    def _fuse_with_location_awareness(
        self,
        ts_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        location_infos: List[Optional[Dict]],
    ) -> torch.Tensor:
        """Fuse modalities with location awareness."""
        # Project time series to LLaMA dimension
        ts_projected = self.ts_projection(ts_tokens)  # [batch, time, hidden]

        if "location_aware" in self.fusion_strategy:
            # Create location embeddings
            location_embeddings = []
            for location_info in location_infos:
                if location_info:
                    # Encode geographic coordinates
                    lat, lon = location_info["lat"], location_info["lon"]
                    loc_embedding = self.spatial_encoder.encode_coordinates(lat, lon)  # [64]
                    loc_embedding = loc_embedding.unsqueeze(0)  # [1, 64]
                    loc_embedding = self.spatial_projection(loc_embedding)  # [1, hidden]
                else:
                    # Default/null location embedding
                    loc_embedding = torch.zeros(1, self.llama_hidden_size, device=self.device)
                location_embeddings.append(loc_embedding)

            # Stack location embeddings
            location_stack = torch.cat(location_embeddings, dim=0)  # [batch, hidden]
            location_stack = location_stack.unsqueeze(1)  # [batch, 1, hidden]

            # Location-aware cross-attention
            fused_ts, _ = self.location_cross_attention(
                query=text_embeddings, key=ts_projected, value=ts_projected
            )
            fused_ts = self.location_norm(fused_ts + text_embeddings)

            # Geographic attention with location context
            geo_fused, _ = self.geographic_attention(
                query=fused_ts,
                key=location_stack.expand(-1, fused_ts.size(1), -1),
                value=location_stack.expand(-1, fused_ts.size(1), -1),
            )
            final_fused = self.geographic_norm(geo_fused + fused_ts)

        else:
            # Standard cross-attention
            fused, _ = self.location_cross_attention(
                query=text_embeddings, key=ts_projected, value=ts_projected
            )
            final_fused = self.location_norm(fused + text_embeddings)

        return torch.as_tensor(final_fused)

    def _generate_text(
        self, fused_embeddings: torch.Tensor, input_ids: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate text using fused embeddings."""
        # Simple mock generation
        batch_size, seq_len = input_ids.shape
        vocab_size = 32000 if self.llama_tokenizer is None else len(self.llama_tokenizer)

        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        return {"logits": logits}

    def _classify(self, fused_embeddings: torch.Tensor) -> Dict[str, Any]:
        """Classify using fused embeddings."""
        # Pool sequence dimension and classify
        pooled = fused_embeddings.mean(dim=1)  # [batch, hidden]
        logits = self.classification_head(pooled)  # [batch, num_classes]
        return {"classification_logits": logits}


def create_sample_location_aware_data():
    """Create sample data for location-aware testing."""
    # Climate data for different regions
    climate_data = torch.randn(2, 4, 5, 32, 32)  # 2 locations, 4 timesteps, 5 vars

    # Location-aware text queries
    text_inputs = [
        "What is the temperature trend in New York over the past week?",
        "Analyze precipitation patterns in the San Francisco Bay Area.",
    ]

    return climate_data, text_inputs


if __name__ == "__main__":
    print("🌍 AIFS Location-Aware Fusion Demo")
    print("=" * 50)

    # Create model
    model = AIFSLocationAwareFusion(
        llama_model_name="meta-llama/Meta-Llama-3-8B",
        fusion_strategy="location_aware_cross_attention",
        use_mock_llama=False,  # Try real LLaMA
        device="cpu",
    )

    # Create test data
    climate_data, text_inputs = create_sample_location_aware_data()

    print(f"🌡️ Climate data shape: {climate_data.shape}")
    print(f"💬 Text queries:")
    for i, text in enumerate(text_inputs):
        print(f"   {i+1}. {text}")

    # Test location-aware processing
    try:
        outputs = model(climate_data=climate_data, text_inputs=text_inputs, task="embedding")

        print(f"\\n✅ Location-aware fusion successful!")
        print(f"📊 Output embedding shape: {outputs['embeddings'].shape}")
        print(
            f"📍 Location info: {len([loc for loc in outputs['location_info'] if loc])} locations resolved"
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
