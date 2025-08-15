"""
Location-Aware Climate Analysis Module

This module implements spatial attention masking and geographic resolution
for location-aware climate queries. It enables the system to understand
questions about specific geographic regions and focus analysis accordingly.

Key Features:
- Geographic entity resolution (text → coordinates)
- Spatial attention masking for climate models
- Location-aware multimodal fusion
- Support for regions, countries, cities, and coordinates
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional geographic database packages
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    warnings.warn("GeoPy not available. Install with: pip install geopy")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("Requests not available. Install with: pip install requests")

# Geographic data - in production, this would come from a proper geographic database
GEOGRAPHIC_DATABASE = {
    # Countries
    "sweden": {"bounds": {"lat_min": 55.3, "lat_max": 69.1, "lon_min": 11.0, "lon_max": 24.2}, "type": "country"},
    "arizona": {"bounds": {"lat_min": 31.3, "lat_max": 37.0, "lon_min": -114.8, "lon_max": -109.0}, "type": "state"},
    "california": {"bounds": {"lat_min": 32.5, "lat_max": 42.0, "lon_min": -124.4, "lon_max": -114.1}, "type": "state"},
    "texas": {"bounds": {"lat_min": 25.8, "lat_max": 36.5, "lon_min": -106.6, "lon_max": -93.5}, "type": "state"},
    "florida": {"bounds": {"lat_min": 24.4, "lat_max": 31.0, "lon_min": -87.6, "lon_max": -80.0}, "type": "state"},
    "norway": {"bounds": {"lat_min": 57.9, "lat_max": 71.2, "lon_min": 4.6, "lon_max": 31.3}, "type": "country"},
    "finland": {"bounds": {"lat_min": 59.8, "lat_max": 70.1, "lon_min": 19.1, "lon_max": 31.6}, "type": "country"},
    "denmark": {"bounds": {"lat_min": 54.5, "lat_max": 57.8, "lon_min": 8.0, "lon_max": 15.2}, "type": "country"},
    "united kingdom": {"bounds": {"lat_min": 49.9, "lat_max": 60.9, "lon_min": -8.6, "lon_max": 1.8}, "type": "country"},
    "germany": {"bounds": {"lat_min": 47.3, "lat_max": 55.1, "lon_min": 5.9, "lon_max": 15.0}, "type": "country"},
    "france": {"bounds": {"lat_min": 41.3, "lat_max": 51.1, "lon_min": -5.1, "lon_max": 9.6}, "type": "country"},
    "spain": {"bounds": {"lat_min": 27.6, "lat_max": 43.8, "lon_min": -18.2, "lon_max": 4.3}, "type": "country"},
    "italy": {"bounds": {"lat_min": 35.5, "lat_max": 47.1, "lon_min": 6.6, "lon_max": 18.5}, "type": "country"},
    "brazil": {"bounds": {"lat_min": -33.7, "lat_max": 5.3, "lon_min": -73.9, "lon_max": -28.8}, "type": "country"},
    "australia": {"bounds": {"lat_min": -43.6, "lat_max": -10.7, "lon_min": 113.3, "lon_max": 153.6}, "type": "country"},
    "india": {"bounds": {"lat_min": 8.1, "lat_max": 37.1, "lon_min": 68.2, "lon_max": 97.4}, "type": "country"},
    "china": {"bounds": {"lat_min": 18.2, "lat_max": 53.6, "lon_min": 73.6, "lon_max": 135.0}, "type": "country"},
    "canada": {"bounds": {"lat_min": 41.7, "lat_max": 83.1, "lon_min": -141.0, "lon_max": -52.6}, "type": "country"},

    # Regions
    "scandinavia": {"bounds": {"lat_min": 54.5, "lat_max": 71.2, "lon_min": 4.6, "lon_max": 31.6}, "type": "region"},
    "mediterranean": {"bounds": {"lat_min": 30.2, "lat_max": 46.3, "lon_min": -6.0, "lon_max": 42.3}, "type": "region"},
    "sahel": {"bounds": {"lat_min": 10.0, "lat_max": 20.0, "lon_min": -20.0, "lon_max": 40.0}, "type": "region"},
    "amazon": {"bounds": {"lat_min": -20.0, "lat_max": 12.0, "lon_min": -82.0, "lon_max": -34.0}, "type": "region"},
    "arctic": {"bounds": {"lat_min": 66.5, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0}, "type": "region"},
    "antarctica": {"bounds": {"lat_min": -90.0, "lat_max": -60.0, "lon_min": -180.0, "lon_max": 180.0}, "type": "region"},
    "sahara": {"bounds": {"lat_min": 15.0, "lat_max": 30.0, "lon_min": -17.0, "lon_max": 51.0}, "type": "region"},
    "himalaya": {"bounds": {"lat_min": 27.0, "lat_max": 35.0, "lon_min": 72.0, "lon_max": 95.0}, "type": "region"},
}

@dataclass
class GeographicLocation:
    """Represents a resolved geographic location with spatial bounds."""
    name: str
    bounds: Dict[str, float]
    center: Dict[str, float]
    location_type: str
    confidence: float

    def __post_init__(self):
        """Calculate center point from bounds if not provided."""
        if not hasattr(self, 'center') or self.center is None:
            self.center = {
                'lat': (self.bounds['lat_min'] + self.bounds['lat_max']) / 2,
                'lon': (self.bounds['lon_min'] + self.bounds['lon_max']) / 2
            }

class GeographicResolver:
    """
    Resolves geographic entities from natural language text.

    Supports multiple backends:
    - 'geopy': Uses GeoPy with Nominatim (OpenStreetMap) - requires internet
    - 'geonames': Uses GeoNames API - requires internet and API key
    - 'local': Uses local database (fallback)

    Converts location names and descriptions into coordinate bounds
    that can be used for spatial attention masking.
    """

    def __init__(self, backend: str = 'auto', geonames_username: str = None):
        """
        Initialize geographic resolver.

        Args:
            backend: 'auto', 'geopy', 'geonames', or 'local'
            geonames_username: Username for GeoNames API (if using geonames backend)
        """
        self.geonames_username = geonames_username

        # Determine best available backend
        if backend == 'auto':
            if GEOPY_AVAILABLE:
                self.backend = 'geopy'
            elif REQUESTS_AVAILABLE and geonames_username:
                self.backend = 'geonames'
            else:
                self.backend = 'local'
        else:
            self.backend = backend

        # Initialize the chosen backend
        self._initialize_backend()

        # Fallback to local database
        self.database = GEOGRAPHIC_DATABASE
        self.coordinate_patterns = [
            r'(\d+\.?\d*)\s*°?\s*([NS])\s*,?\s*(\d+\.?\d*)\s*°?\s*([EW])',  # 40.7°N, 74.0°W
            r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',  # 40.7, -74.0
        ]

        # Cache for resolved locations
        self._cache = {}

    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == 'geopy':
            if not GEOPY_AVAILABLE:
                warnings.warn("GeoPy not available, falling back to local database")
                self.backend = 'local'
                return

            # Initialize Nominatim geocoder
            self.geocoder = Nominatim(
                user_agent="prithvi-climate-analysis",
                timeout=10
            )
            print(f"🌍 Using GeoPy/Nominatim geocoder for geographic resolution")

        elif self.backend == 'geonames':
            if not REQUESTS_AVAILABLE or not self.geonames_username:
                warnings.warn("GeoNames requires requests and username, falling back to local database")
                self.backend = 'local'
                return

            self.geonames_base_url = "http://api.geonames.org"
            print(f"🌍 Using GeoNames API for geographic resolution")

        else:  # local
            print(f"🌍 Using local geographic database")

    def resolve_location(self, location_text: str) -> Optional[GeographicLocation]:
        """
        Resolve a location string to geographic bounds.

        Args:
            location_text: Location name or coordinate string

        Returns:
            GeographicLocation object or None if not found
        """
        # Check cache first
        if location_text in self._cache:
            return self._cache[location_text]

        location_lower = location_text.lower().strip()

        # Try coordinate parsing first (works for all backends)
        coord_result = self._parse_coordinates(location_text)
        if coord_result:
            self._cache[location_text] = coord_result
            return coord_result

        # Try the selected backend
        result = None

        if self.backend == 'geopy':
            result = self._resolve_with_geopy(location_text)
        elif self.backend == 'geonames':
            result = self._resolve_with_geonames(location_text)

        # Fallback to local database if backend fails
        if not result:
            result = self._resolve_with_local_db(location_text)

        # Cache the result
        if result:
            self._cache[location_text] = result

        return result

    def _resolve_with_geopy(self, location_text: str) -> Optional[GeographicLocation]:
        """Resolve location using GeoPy/Nominatim."""
        try:
            # Search for the location
            location = self.geocoder.geocode(
                location_text,
                exactly_one=True,
                addressdetails=True,
                extratags=True
            )

            if not location:
                return None

            # Get detailed information
            raw = location.raw

            # Determine location type
            location_type = self._determine_type_from_geopy(raw)

            # Create bounding box
            if 'boundingbox' in raw:
                # Nominatim provides bounding box: [south, north, west, east]
                bbox = raw['boundingbox']
                bounds = {
                    'lat_min': float(bbox[0]),
                    'lat_max': float(bbox[1]),
                    'lon_min': float(bbox[2]),
                    'lon_max': float(bbox[3])
                }
            else:
                # Create small bounding box around point
                margin = self._get_margin_for_type(location_type)
                bounds = {
                    'lat_min': location.latitude - margin,
                    'lat_max': location.latitude + margin,
                    'lon_min': location.longitude - margin,
                    'lon_max': location.longitude + margin
                }

            return GeographicLocation(
                name=location.address,
                bounds=bounds,
                center={'lat': location.latitude, 'lon': location.longitude},
                location_type=location_type,
                confidence=0.9
            )

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            warnings.warn(f"Geocoding failed for '{location_text}': {e}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error in geocoding '{location_text}': {e}")
            return None

    def _resolve_with_geonames(self, location_text: str) -> Optional[GeographicLocation]:
        """Resolve location using GeoNames API."""
        try:
            # Search for places
            search_url = f"{self.geonames_base_url}/searchJSON"
            params = {
                'q': location_text,
                'maxRows': 1,
                'username': self.geonames_username,
                'style': 'full'
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get('geonames'):
                return None

            place = data['geonames'][0]

            # Get bounding box if available
            if 'bbox' in place:
                bbox = place['bbox']
                bounds = {
                    'lat_min': bbox['south'],
                    'lat_max': bbox['north'],
                    'lon_min': bbox['west'],
                    'lon_max': bbox['east']
                }
            else:
                # Create bounding box based on feature class
                margin = self._get_margin_for_geonames_feature(place.get('fclName', ''))
                lat, lon = float(place['lat']), float(place['lng'])
                bounds = {
                    'lat_min': lat - margin,
                    'lat_max': lat + margin,
                    'lon_min': lon - margin,
                    'lon_max': lon + margin
                }

            return GeographicLocation(
                name=place.get('name', location_text),
                bounds=bounds,
                center={'lat': float(place['lat']), 'lon': float(place['lng'])},
                location_type=self._geonames_to_type(place.get('fclName', '')),
                confidence=0.85
            )

        except requests.RequestException as e:
            warnings.warn(f"GeoNames API request failed for '{location_text}': {e}")
            return None
        except Exception as e:
            warnings.warn(f"Unexpected error in GeoNames lookup '{location_text}': {e}")
            return None

    def _resolve_with_local_db(self, location_text: str) -> Optional[GeographicLocation]:
        """Resolve location using local database (original implementation)."""
        location_lower = location_text.lower().strip()

        # Try exact match
        if location_lower in self.database:
            data = self.database[location_lower]
            return GeographicLocation(
                name=location_text,
                bounds=data["bounds"],
                center=None,  # Will be calculated in __post_init__
                location_type=data["type"],
                confidence=0.9
            )

        # Try partial matching
        for db_location, data in self.database.items():
            if db_location in location_lower or location_lower in db_location:
                return GeographicLocation(
                    name=location_text,
                    bounds=data["bounds"],
                    center=None,
                    location_type=data["type"],
                    confidence=0.7
                )

        return None

    def _determine_type_from_geopy(self, raw_data: dict) -> str:
        """Determine location type from GeoPy raw data."""
        address = raw_data.get('address', {})
        osm_type = raw_data.get('osm_type', '')
        place_type = raw_data.get('type', '')

        # Check for country
        if address.get('country') and not any(k in address for k in ['state', 'city', 'town', 'village']):
            return 'country'

        # Check for state/province
        if address.get('state') and not any(k in address for k in ['city', 'town', 'village']):
            return 'state'

        # Check for city/town
        if any(k in address for k in ['city', 'town', 'village']):
            return 'city'

        # Check for regions based on place type
        if place_type in ['sea', 'ocean', 'bay', 'gulf']:
            return 'region'

        return 'place'

    def _geonames_to_type(self, fcl_name: str) -> str:
        """Convert GeoNames feature class to our location type."""
        fcl_lower = fcl_name.lower()

        if 'country' in fcl_lower:
            return 'country'
        elif any(word in fcl_lower for word in ['state', 'province', 'region']):
            return 'state'
        elif any(word in fcl_lower for word in ['city', 'town', 'village']):
            return 'city'
        elif any(word in fcl_lower for word in ['mountain', 'hill', 'forest', 'desert', 'sea', 'ocean']):
            return 'region'
        else:
            return 'place'

    def _get_margin_for_type(self, location_type: str) -> float:
        """Get appropriate margin based on location type."""
        margins = {
            'coordinate': 0.1,
            'city': 0.5,
            'state': 2.0,
            'country': 5.0,
            'region': 10.0,
            'place': 1.0
        }
        return margins.get(location_type, 1.0)

    def _get_margin_for_geonames_feature(self, feature_name: str) -> float:
        """Get margin based on GeoNames feature class."""
        feature_lower = feature_name.lower()

        if 'country' in feature_lower:
            return 5.0
        elif any(word in feature_lower for word in ['state', 'province']):
            return 2.0
        elif any(word in feature_lower for word in ['city', 'town']):
            return 0.5
        else:
            return 1.0

    def _parse_coordinates(self, coord_text: str) -> Optional[GeographicLocation]:
        """Parse coordinate strings into geographic locations."""
        for pattern in self.coordinate_patterns:
            match = re.search(pattern, coord_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 4:  # Lat/Lon with N/S/E/W
                    lat = float(match.group(1))
                    lat_dir = match.group(2).upper()
                    lon = float(match.group(3))
                    lon_dir = match.group(4).upper()

                    if lat_dir == 'S':
                        lat = -lat
                    if lon_dir == 'W':
                        lon = -lon

                elif len(match.groups()) == 2:  # Simple lat, lon
                    lat = float(match.group(1))
                    lon = float(match.group(2))

                # Create small bounding box around point
                margin = 0.5  # 0.5 degree margin
                bounds = {
                    "lat_min": lat - margin,
                    "lat_max": lat + margin,
                    "lon_min": lon - margin,
                    "lon_max": lon + margin
                }

                return GeographicLocation(
                    name=coord_text,
                    bounds=bounds,
                    center={"lat": lat, "lon": lon},
                    location_type="coordinate",
                    confidence=0.95
                )

        return None

    def extract_locations(self, text: str) -> List[str]:
        """
        Extract potential location names from text.

        Args:
            text: Input text containing location references

        Returns:
            List of potential location names
        """
        # Simple pattern matching - in production, use NER models
        locations = []

        # Check for coordinate patterns first
        for pattern in self.coordinate_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                locations.append(match.group(0))

        # Check for known geographic entities
        text_lower = text.lower()
        for location in self.database.keys():
            if location in text_lower:
                locations.append(location)

        return locations

class SpatialCropper:
    """
    Creates spatial attention masks for focusing climate models on specific regions.

    This class handles the conversion from geographic bounds to grid indices
    and creates attention masks compatible with the Prithvi-WxC architecture.
    """

    def __init__(self, grid_shape: Tuple[int, int] = (360, 576)):
        """
        Initialize spatial cropper.

        Args:
            grid_shape: (n_lats, n_lons) - MERRA-2 grid dimensions
        """
        self.n_lats, self.n_lons = grid_shape

        # MERRA-2 grid: 0.5° × 0.625° resolution
        # Latitudes: -89.75 to 89.75 (360 points)
        # Longitudes: -180 to 179.375 (576 points)
        self.lat_resolution = 180.0 / self.n_lats  # 0.5 degrees
        self.lon_resolution = 360.0 / self.n_lons  # 0.625 degrees

        self.lat_min_global = -90.0 + self.lat_resolution / 2
        self.lat_max_global = 90.0 - self.lat_resolution / 2
        self.lon_min_global = -180.0 + self.lon_resolution / 2
        self.lon_max_global = 180.0 - self.lon_resolution / 2

    def lat_to_grid_index(self, lat: float) -> int:
        """Convert latitude to grid index."""
        # Clamp to valid range
        lat = max(min(lat, self.lat_max_global), self.lat_min_global)
        # Convert to index (0 = southernmost, 359 = northernmost)
        index = int((lat - self.lat_min_global) / self.lat_resolution)
        return max(0, min(index, self.n_lats - 1))

    def lon_to_grid_index(self, lon: float) -> int:
        """Convert longitude to grid index."""
        # Handle longitude wrapping, but treat 180.0 as valid (date line)
        while lon < -180:
            lon += 360
        while lon > 180:
            lon -= 360

        # Special case: longitude 180.0 (date line) should map to the last index
        if lon == 180.0:
            return self.n_lons - 1

        # Convert to index (0 = westernmost, 575 = easternmost)
        index = int((lon - self.lon_min_global) / self.lon_resolution)
        return max(0, min(index, self.n_lons - 1))

    def create_location_mask(
        self,
        location: GeographicLocation,
        mask_type: str = "gaussian",
        focus_strength: float = 3.0
    ) -> torch.Tensor:
        """
        Create spatial attention mask for a geographic location.

        Args:
            location: GeographicLocation with bounds
            mask_type: Type of mask ('binary', 'gaussian', 'cosine')
            focus_strength: Strength of attention focus (higher = more focused)

        Returns:
            Tensor of shape [n_lats, n_lons] with attention weights
        """
        bounds = location.bounds

        # Convert bounds to grid indices
        lat_min_idx = self.lat_to_grid_index(bounds["lat_min"])
        lat_max_idx = self.lat_to_grid_index(bounds["lat_max"])
        lon_min_idx = self.lon_to_grid_index(bounds["lon_min"])
        lon_max_idx = self.lon_to_grid_index(bounds["lon_max"])

        # Handle longitude wrapping
        if bounds["lon_min"] > bounds["lon_max"]:  # Crosses dateline
            mask1 = self._create_mask_region(
                lat_min_idx, lat_max_idx, lon_min_idx, self.n_lons - 1,
                mask_type, focus_strength
            )
            mask2 = self._create_mask_region(
                lat_min_idx, lat_max_idx, 0, lon_max_idx,
                mask_type, focus_strength
            )
            return torch.maximum(mask1, mask2)
        else:
            return self._create_mask_region(
                lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx,
                mask_type, focus_strength
            )

    def _create_mask_region(
        self,
        lat_min_idx: int,
        lat_max_idx: int,
        lon_min_idx: int,
        lon_max_idx: int,
        mask_type: str,
        focus_strength: float
    ) -> torch.Tensor:
        """Create mask for a specific grid region."""
        mask = torch.zeros(self.n_lats, self.n_lons)

        if mask_type == "binary":
            mask[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1] = 1.0

        elif mask_type == "gaussian":
            # Create Gaussian mask centered on region
            lat_center = (lat_min_idx + lat_max_idx) / 2
            lon_center = (lon_min_idx + lon_max_idx) / 2
            lat_std = (lat_max_idx - lat_min_idx) / focus_strength
            lon_std = (lon_max_idx - lon_min_idx) / focus_strength

            lat_indices = torch.arange(self.n_lats).float()
            lon_indices = torch.arange(self.n_lons).float()

            lat_weights = torch.exp(-0.5 * ((lat_indices - lat_center) / lat_std) ** 2)
            lon_weights = torch.exp(-0.5 * ((lon_indices - lon_center) / lon_std) ** 2)

            mask = lat_weights[:, None] * lon_weights[None, :]

        elif mask_type == "cosine":
            # Create cosine taper mask
            lat_center = (lat_min_idx + lat_max_idx) / 2
            lon_center = (lon_min_idx + lon_max_idx) / 2
            lat_radius = (lat_max_idx - lat_min_idx) / 2 * focus_strength
            lon_radius = (lon_max_idx - lon_min_idx) / 2 * focus_strength

            for i in range(self.n_lats):
                for j in range(self.n_lons):
                    lat_dist = abs(i - lat_center) / lat_radius
                    lon_dist = abs(j - lon_center) / lon_radius

                    if lat_dist <= 1.0 and lon_dist <= 1.0:
                        lat_weight = (1 + torch.cos(torch.pi * torch.tensor(lat_dist))) / 2
                        lon_weight = (1 + torch.cos(torch.pi * torch.tensor(lon_dist))) / 2
                        mask[i, j] = lat_weight * lon_weight

        return mask

class LocationAwareAttention(nn.Module):
    """
    Location-aware attention module that incorporates spatial masks.

    This module modifies the standard transformer attention to focus
    on specific geographic regions based on spatial masks.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.spatial_gate = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional spatial masking.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            spatial_mask: Spatial attention mask [batch, seq_len] or [seq_len]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, C = x.shape

        # Standard multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply spatial mask if provided
        if spatial_mask is not None:
            if spatial_mask.dim() == 1:
                spatial_mask = spatial_mask.unsqueeze(0)  # Add batch dimension
            if spatial_mask.dim() == 2:
                spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1)  # Add head and key dimensions

            # Apply mask to attention scores
            spatial_weights = self.spatial_gate(spatial_mask.unsqueeze(-1)).squeeze(-1)
            attn = attn * spatial_weights

        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

def demo_location_resolution():
    """Demonstrate location resolution capabilities with different backends."""
    print("🌍 Geographic Location Resolution Demo\n")

    test_queries = [
        "What crops will be viable in Sweden by 2050?",
        "How sustainable will Arizona be by 2100?",
        "Climate trends in the Mediterranean region",
        "Tornado frequency changes in 40.7°N, 74.0°W",
        "Arctic ice melting patterns"
    ]

    # Test different backends
    backends = ['local']
    if GEOPY_AVAILABLE:
        backends.append('geopy')

    for backend in backends:
        print(f"🔧 Testing {backend.upper()} backend:")
        resolver = GeographicResolver(backend=backend)

        for query in test_queries:
            print(f"  Query: {query}")
            locations = resolver.extract_locations(query)

            for loc_text in locations:
                location = resolver.resolve_location(loc_text)
                if location:
                    print(f"    ✓ Found: {location.name}")
                    print(f"      Type: {location.location_type}")
                    print(f"      Bounds: {location.bounds}")
                    print(f"      Center: {location.center}")
                    print(f"      Confidence: {location.confidence}")
                else:
                    print(f"    ✗ Could not resolve: {loc_text}")
            print()

        print("-" * 50)

    # Show backend capabilities
    print("\n📊 Backend Comparison:")
    print("LOCAL: Fast, offline, limited coverage")
    if GEOPY_AVAILABLE:
        print("GEOPY: Comprehensive, requires internet, free (Nominatim)")
    else:
        print("GEOPY: Not installed (pip install geopy)")

    if REQUESTS_AVAILABLE:
        print("GEONAMES: Comprehensive, requires internet + API key")
    else:
        print("REQUESTS: Not installed (pip install requests)")

def demo_geographic_backends():
    """Demonstrate different geographic backends in detail."""
    print("🗺️  Geographic Backend Demonstration\n")

    test_location = "Stockholm, Sweden"

    print(f"Testing resolution of: '{test_location}'\n")

    # Test local backend
    print("1. LOCAL Backend (Built-in Database):")
    local_resolver = GeographicResolver(backend='local')
    result = local_resolver.resolve_location(test_location)
    if result:
        print(f"   ✓ Resolved: {result.name} ({result.location_type})")
        print(f"   📍 Center: {result.center['lat']:.2f}°, {result.center['lon']:.2f}°")
    else:
        print(f"   ✗ Not found in local database")
    print()

    # Test GeoPy backend
    if GEOPY_AVAILABLE:
        print("2. GEOPY Backend (OpenStreetMap/Nominatim):")
        try:
            geopy_resolver = GeographicResolver(backend='geopy')
            result = geopy_resolver.resolve_location(test_location)
            if result:
                print(f"   ✓ Resolved: {result.name}")
                print(f"   📍 Center: {result.center['lat']:.2f}°, {result.center['lon']:.2f}°")
                print(f"   📦 Bounding box: {result.bounds}")
                print(f"   🎯 Type: {result.location_type}")
                print(f"   🔍 Confidence: {result.confidence}")
            else:
                print(f"   ✗ Not found via GeoPy")
        except Exception as e:
            print(f"   ⚠️  GeoPy error: {e}")
    else:
        print("2. GEOPY Backend: Not available (install with: pip install geopy)")
    print()

    # Show installation instructions
    print("📦 Installation Instructions:")
    print("   pip install geopy requests")
    print("   # For GeoNames: register at geonames.org for free username")
    print("   # For enhanced features: pip install shapely geopandas folium")

def demo_spatial_masking():
    """Demonstrate spatial mask creation."""
    cropper = SpatialCropper()
    resolver = GeographicResolver()

    # Test with Sweden
    sweden = resolver.resolve_location("sweden")
    if sweden:
        print("🗺️  Spatial Mask Demo - Sweden\n")

        # Create different types of masks
        binary_mask = cropper.create_location_mask(sweden, "binary")
        gaussian_mask = cropper.create_location_mask(sweden, "gaussian")

        print(f"Binary mask sum: {binary_mask.sum():.0f} pixels")
        print(f"Gaussian mask sum: {gaussian_mask.sum():.2f}")
        print(f"Grid shape: {binary_mask.shape}")
        print(f"Sweden coverage: {binary_mask.sum() / binary_mask.numel() * 100:.2f}% of global grid")

if __name__ == "__main__":
    demo_location_resolution()
    print("\n" + "="*60 + "\n")
    demo_geographic_backends()
    print("\n" + "="*60 + "\n")
    demo_spatial_masking()
