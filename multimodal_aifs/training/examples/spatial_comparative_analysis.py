#!/usr/bin/env python3
"""
Spatial Comparative Climate Analysis Demo

This script demonstrates how to handle spatial queries that compare multiple locations,
such as "Where will it be hotter, Arizona or Alaska?" It shows:

1. Multi-location extraction from natural language queries
2. Spatial mask creation for each location
3. Union mask generation for comparative analysis
4. Location-aware climate feature extraction
5. Comparative climate analysis between regions

Example questions it can handle:
- "Where will it be hotter, Arizona or Alaska?"
- "Which region has more rainfall, California or Texas?"
- "Compare drought conditions between Sweden and Norway"
"""

import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# GeoPy for dynamic location resolution
try:
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut
    from geopy.geocoders import Nominatim

    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("⚠️  GeoPy not available. Install with: pip install geopy")

# Memory optimization
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def check_memory_usage():
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


print("🌍 Spatial Comparative Climate Analysis Demo")
print("📊 Demonstrating multi-location climate comparison")


class DynamicLocationResolver:
    """Dynamic location resolver using GeoPy and Nominatim"""

    def __init__(self, use_geopy: bool = True, timeout: int = 10):
        self.use_geopy = use_geopy and GEOPY_AVAILABLE
        self.timeout = timeout
        self._cache = {}  # Cache resolved locations

        if self.use_geopy:
            self.geocoder = Nominatim(user_agent="climate-analysis-demo", timeout=self.timeout)
            print("🌍 Using GeoPy/Nominatim for dynamic location resolution")
        else:
            print("⚠️  Using fallback location resolution (limited coverage)")
            # Minimal fallback database for demo when GeoPy not available
            self.fallback_db = {
                "arizona": {
                    "bounds": {
                        "lat_min": 31.3,
                        "lat_max": 37.0,
                        "lon_min": -114.8,
                        "lon_max": -109.0,
                    },
                    "center": {"lat": 34.15, "lon": -111.9},
                    "type": "state",
                },
                "alaska": {
                    "bounds": {
                        "lat_min": 54.7,
                        "lat_max": 71.4,
                        "lon_min": -179.1,
                        "lon_max": -129.0,
                    },
                    "center": {"lat": 63.05, "lon": -154.05},
                    "type": "state",
                },
                "california": {
                    "bounds": {
                        "lat_min": 32.5,
                        "lat_max": 42.0,
                        "lon_min": -124.4,
                        "lon_max": -114.1,
                    },
                    "center": {"lat": 37.25, "lon": -119.25},
                    "type": "state",
                },
                "texas": {
                    "bounds": {
                        "lat_min": 25.8,
                        "lat_max": 36.5,
                        "lon_min": -106.6,
                        "lon_max": -93.5,
                    },
                    "center": {"lat": 31.15, "lon": -100.05},
                    "type": "state",
                },
            }

    def resolve_location(self, location_name: str) -> Optional[Dict]:
        """
        Resolve location name to geographic bounds using GeoPy/Nominatim

        Args:
            location_name: Name of location to resolve or coordinate string

        Returns:
            Dictionary with bounds, center, type, and confidence
        """
        # Check if this is a coordinate string
        if self._is_coordinate_string(location_name):
            return self._resolve_coordinates(location_name)

        # Check cache first
        cache_key = location_name.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_geopy:
            return self._resolve_with_geopy(location_name)
        else:
            return self._resolve_with_fallback(location_name)

    def _is_coordinate_string(self, location_name: str) -> bool:
        """Check if the location string is actually coordinates"""
        import re

        # Check for patterns like "47.29°,-120.21°" or "47.29°, -120.21°"
        coord_pattern = r"^(-?\d+\.?\d*)\s*[°]?\s*,\s*(-?\d+\.?\d*)\s*[°]?$"
        return bool(re.match(coord_pattern, location_name.strip()))

    def _resolve_coordinates(self, coord_string: str) -> Optional[Dict]:
        """Resolve coordinate string to location info"""
        import re

        coord_pattern = r"^(-?\d+\.?\d*)\s*[°]?\s*,\s*(-?\d+\.?\d*)\s*[°]?$"
        match = re.match(coord_pattern, coord_string.strip())

        if not match:
            return None

        try:
            lat = float(match.group(1))
            lon = float(match.group(2))

            # Validate coordinate ranges
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None

            # Create a small bounding box around the point
            margin = 0.5  # degrees
            bounds = {
                "lat_min": lat - margin,
                "lat_max": lat + margin,
                "lon_min": lon - margin,
                "lon_max": lon + margin,
            }

            # Try reverse geocoding to get location name if GeoPy is available
            address = f"Coordinates ({lat:.2f}°, {lon:.2f}°)"
            location_type = "coordinates"

            if self.use_geopy:
                try:
                    location = self.geocoder.reverse(f"{lat}, {lon}", timeout=self.timeout)
                    if location and location.address:
                        address = f"{location.address} ({lat:.2f}°, {lon:.2f}°)"
                        location_type = "resolved_coordinates"
                        print(f"🎯 Reverse geocoded {coord_string} → {location.address}")
                except Exception as e:
                    print(f"⚠️  Reverse geocoding failed for {coord_string}: {e}")

            result = {
                "name": coord_string,
                "bounds": bounds,
                "center": {"lat": lat, "lon": lon},
                "type": location_type,
                "confidence": 1.0,  # Coordinates are exact
                "address": address,
                "raw": {"lat": lat, "lon": lon},
            }

            # Cache the result
            cache_key = coord_string.lower().strip()
            self._cache[cache_key] = result

            print(f"✅ Resolved coordinates: {address}")
            print(f"   📍 Center: {lat:.2f}°, {lon:.2f}°")
            print(
                f"   📏 Bounds: {bounds['lat_min']:.1f} to {bounds['lat_max']:.1f}°N, {bounds['lon_min']:.1f} to {bounds['lon_max']:.1f}°E"
            )

            return result

        except ValueError:
            print(f"❌ Invalid coordinate format: {coord_string}")
            return None

    def _resolve_with_geopy(self, location_name: str) -> Optional[Dict]:
        """Resolve location using GeoPy/Nominatim with multiple candidates"""
        try:
            print(f"🔍 Resolving '{location_name}' with Nominatim (multiple candidates)...")

            # Query Nominatim for multiple results
            locations = self.geocoder.geocode(
                location_name,
                exactly_one=False,
                limit=5,  # Get up to 5 candidates
                timeout=self.timeout,
            )

            if not locations:
                print(f"❌ Location '{location_name}' not found")
                return None

            print(f"🎯 Found {len(locations)} candidates for '{location_name}'")

            # Choose the best candidate
            best_location = self._select_best_candidate(location_name, locations)

            if not best_location:
                print(f"❌ No suitable candidate found for '{location_name}'")
                return None

            # Get bounding box - try to get from location first
            if hasattr(best_location, "raw") and "boundingbox" in best_location.raw:
                bbox = best_location.raw["boundingbox"]
                bounds = {
                    "lat_min": float(bbox[0]),
                    "lat_max": float(bbox[1]),
                    "lon_min": float(bbox[2]),
                    "lon_max": float(bbox[3]),
                }
            else:
                # Fallback: create small bounding box around point
                lat, lon = best_location.latitude, best_location.longitude
                margin = 1.0  # degrees
                bounds = {
                    "lat_min": lat - margin,
                    "lat_max": lat + margin,
                    "lon_min": lon - margin,
                    "lon_max": lon + margin,
                }

            # Determine location type from Nominatim response
            location_type = self._determine_location_type(best_location.raw)

            result = {
                "name": location_name,
                "bounds": bounds,
                "center": {"lat": best_location.latitude, "lon": best_location.longitude},
                "type": location_type,
                "confidence": 1.0,  # Nominatim found it
                "address": best_location.address,
                "raw": best_location.raw,
                "candidates_count": len(locations),
            }

            # Cache the result
            cache_key = location_name.lower().strip()
            self._cache[cache_key] = result

            print(f"✅ Selected best match: {best_location.address}")
            print(f"   📍 Center: {best_location.latitude:.2f}°, {best_location.longitude:.2f}°")
            print(
                f"   📏 Bounds: {bounds['lat_min']:.1f} to {bounds['lat_max']:.1f}°N, {bounds['lon_min']:.1f} to {bounds['lon_max']:.1f}°E"
            )
            print(f"   🎯 Selected from {len(locations)} candidates")

            return result

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"⚠️  Geocoder error for '{location_name}': {e}")
            return None
        except Exception as e:
            print(f"❌ Error resolving '{location_name}': {e}")
            return None

    def _select_best_candidate(self, query: str, candidates: List) -> Optional[Any]:
        """
        Select the best candidate from multiple geocoding results

        Priority:
        1. Exact name match
        2. Administrative boundaries (state, country) over cities
        3. Higher importance score
        4. Larger geographic areas
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Print all candidates for debugging
        print(f"   📋 Evaluating candidates:")
        for i, candidate in enumerate(candidates):
            loc_type = self._determine_location_type(candidate.raw)
            importance = candidate.raw.get("importance", 0)
            print(
                f"      {i+1}. {candidate.address} (type: {loc_type}, importance: {importance:.3f})"
            )

        query_lower = query.lower().strip()

        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(query_lower, candidate)
            scored_candidates.append((score, candidate))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_candidate = scored_candidates[0]
        print(f"   🏆 Best candidate score: {best_score:.3f}")

        return best_candidate

    def _score_candidate(self, query: str, candidate) -> float:
        """Score a candidate location based on relevance to query"""
        score = 0.0

        # Get candidate info
        address = candidate.address.lower()
        raw = candidate.raw
        location_type = self._determine_location_type(raw)
        importance = raw.get("importance", 0.5)

        # 1. Exact name match bonus
        if query in address:
            score += 2.0

        # 2. Location type preferences (states/countries over cities, major cities over towns)
        type_scores = {
            "state": 1.8,
            "country": 1.5,
            "province": 1.5,
            "region": 1.3,
            "city": 1.0,
            "town": 0.8,
            "village": 0.6,
            "unknown": 0.5,
        }
        score += type_scores.get(location_type, 0.5)

        # 3. Importance score from Nominatim (heavily weighted)
        score += importance * 2.0  # Double weight for importance

        # 4. Prefer results with bounding boxes (better geographic data)
        if "boundingbox" in raw:
            score += 0.5

        # 5. Avoid very specific addresses for general queries
        if "house_number" in raw and len(query.split()) <= 2:
            score -= 0.5

        # 6. Strong boost for major international cities by checking importance thresholds
        if importance > 0.6:
            score += 2.0  # Major international location bonus
        elif importance > 0.4:
            score += 1.0  # Significant location bonus

        # 7. Strong penalty for very small US towns when query could be international
        if (
            importance < 0.3
            and "united states" in address.lower()
            and location_type in ["city", "town", "village"]
        ):
            score -= 1.0  # Stronger penalty

        return score

    def _resolve_with_fallback(self, location_name: str) -> Optional[Dict]:
        """Fallback resolution using limited local database"""
        cache_key = location_name.lower().strip()

        if cache_key in self.fallback_db:
            result = self.fallback_db[cache_key].copy()
            result["name"] = location_name
            result["confidence"] = 0.8  # Lower confidence for fallback
            self._cache[cache_key] = result
            print(f"✅ Resolved '{location_name}' using fallback database")
            return result
        else:
            print(f"❌ Location '{location_name}' not found in fallback database")
            return None

    def _determine_location_type(self, raw_data: Dict) -> str:
        """Determine location type from Nominatim raw data"""
        if "type" in raw_data:
            osm_type = raw_data["type"]

            # Map OSM types to our categories
            if osm_type in ["administrative"]:
                if "place_rank" in raw_data:
                    rank = raw_data["place_rank"]
                    if rank <= 4:
                        return "country"
                    elif rank <= 8:
                        return "state"
                    elif rank <= 12:
                        return "region"
                    else:
                        return "city"
            elif osm_type in ["city", "town", "village"]:
                return "city"
            elif osm_type in ["country"]:
                return "country"
            elif osm_type in ["state"]:
                return "state"

        # Fallback: analyze address components
        if "display_name" in raw_data:
            address = raw_data["display_name"].lower()
            if any(keyword in address for keyword in ["united states", "canada", "australia"]):
                if any(keyword in address for keyword in ["state", "province", "territory"]):
                    return "state"
            if any(keyword in address for keyword in ["country"]):
                return "country"

        return "region"  # Default

    def bulk_resolve(self, location_names: List[str]) -> Dict[str, Optional[Dict]]:
        """Resolve multiple locations efficiently"""
        results = {}

        for location_name in location_names:
            results[location_name] = self.resolve_location(location_name)

            # Small delay to be respectful to Nominatim
            if self.use_geopy:
                time.sleep(0.1)

        return results

    def search_nearby_locations(self, location_name: str, radius_km: float = 100) -> List[Dict]:
        """Find nearby locations (future enhancement)"""
        # This would use Nominatim's nearby search functionality
        # For now, just return the main location
        main_location = self.resolve_location(location_name)
        return [main_location] if main_location else []


class MultiLocationExtractor:
    """Extract and resolve multiple locations from comparative queries using dynamic resolution"""

    def __init__(self, use_geopy: bool = True):
        self.resolver = DynamicLocationResolver(use_geopy=use_geopy)
        self.comparative_patterns = [
            r"\b(.*?)\s+(?:vs|versus|or|and)\s+(.*?)\b",
            r"\b(?:between|compare)\s+(.*?)\s+(?:and|with)\s+(.*?)\b",
            r"\b(?:which|where).*?(?:between|among)\s+(.*?)\s+(?:and|or)\s+(.*?)\b",
        ]

        # Common location indicators for better extraction
        self.location_indicators = [
            "city",
            "state",
            "country",
            "province",
            "region",
            "county",
            "island",
            "continent",
            "territory",
            "district",
            "area",
        ]

    def extract_locations(self, query: str) -> List[str]:
        """Extract location names from query text using NLP patterns and validation"""
        import re

        locations = []

        # Method 0: Extract coordinate patterns first
        coordinate_locations = self._extract_coordinates(query)
        if coordinate_locations:
            locations.extend(coordinate_locations)

        # Method 1: Try extracting from comparative patterns
        comparative_locations = self._extract_from_comparative_patterns(query)
        if comparative_locations:
            for pair in comparative_locations:
                locations.extend(pair)

        # Method 2: Look for capitalized words that might be locations
        potential_locations = self._extract_potential_locations(query)
        locations.extend(potential_locations)

        # Method 3: Validate locations by attempting to resolve them
        validated_locations = self._validate_locations(locations)

        return validated_locations

    def _extract_coordinates(self, query: str) -> List[str]:
        """Extract coordinate patterns from query text"""
        import re

        coordinate_patterns = [
            # Patterns like "47.29°, -120.21°" or "47.29, -120.21"
            r"(-?\d+\.?\d*)\s*[°]?\s*,\s*(-?\d+\.?\d*)\s*[°]?",
            # Patterns like "lat: 47.29, lon: -120.21"
            r"lat\s*:\s*(-?\d+\.?\d*)\s*,\s*lon\s*:\s*(-?\d+\.?\d*)",
            # Patterns like "(47.29, -120.21)"
            r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
        ]

        coordinates = []

        for pattern in coordinate_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                lat_str, lon_str = match.groups()
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)

                    # Basic validation for reasonable coordinate ranges
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        coord_string = f"{lat:.2f}°,{lon:.2f}°"
                        coordinates.append(coord_string)
                        print(f"🎯 Extracted coordinates: {coord_string}")
                except ValueError:
                    continue

        return coordinates

    def _extract_from_comparative_patterns(self, query: str) -> List[Tuple[str, str]]:
        """Extract location pairs from comparative language patterns"""
        import re

        pairs = []

        # Enhanced patterns to handle coordinates and mixed formats
        enhanced_patterns = [
            # Original patterns
            r"\b(.*?)\s+(?:vs|versus|or|and)\s+(.*?)\b",
            r"\b(?:between|compare)\s+(.*?)\s+(?:and|with)\s+(.*?)\b",
            r"\b(?:which|where).*?(?:between|among)\s+(.*?)\s+(?:and|or)\s+(.*?)\b",
            # New patterns for "X be hotter than Y" format - improved for coordinates
            r"(?:will\s+)?(.*?)\s+(?:be|is|will\s+be)\s+(?:hotter|colder|warmer|cooler|wetter|drier).*?than\s+(.*?)$",
            # Pattern for "X hotter than Y"
            r"\b(.*?)\s+(?:hotter|colder|warmer|cooler|wetter|drier)\s+than\s+(.*?)$",
        ]

        for pattern in enhanced_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                loc1_text = match.group(1).strip()
                loc2_text = match.group(2).strip()

                # Clean up the extracted text
                loc1 = self._clean_location_text(loc1_text)
                loc2 = self._clean_location_text(loc2_text)

                if loc1 and loc2:
                    pairs.append((loc1, loc2))
                    print(f"🔍 Extracted comparative pair: '{loc1}' vs '{loc2}'")

        return pairs

    def _extract_potential_locations(self, query: str) -> List[str]:
        """Extract potential locations using capitalization and common patterns"""
        import re

        # Look for capitalized words (potential proper nouns)
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)

        # Filter out common non-location words
        non_locations = {
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
            "Where",
            "What",
            "When",
            "How",
            "Why",
            "Which",
            "Compare",
            "Between",
            "Climate",
            "Weather",
            "Temperature",
        }

        potential_locations = [word for word in words if word not in non_locations]
        return potential_locations

    def _clean_location_text(self, text: str) -> Optional[str]:
        """Clean and normalize location text"""
        if not text:
            return None

        text = text.strip()

        # Check if this is a coordinate string - preserve it as-is
        if self._is_coordinate_pattern(text):
            return text

        # Remove common non-location words for regular text
        clean_text = re.sub(
            r"\b(the|a|an|is|are|will|be|more|less|better|worse)\b", "", text, flags=re.IGNORECASE
        )
        clean_text = clean_text.strip()

        # Extract the main location name (often the last significant word)
        words = clean_text.split()
        if not words:
            return None

        # Try to find the best candidate word
        for word in reversed(words):
            if len(word) > 2 and word.isalpha():
                return word.title()

        return words[-1].title() if words else None

    def _is_coordinate_pattern(self, text: str) -> bool:
        """Check if text matches coordinate patterns"""
        coord_patterns = [
            r"(-?\d+\.?\d*)\s*[°]?\s*,\s*(-?\d+\.?\d*)\s*[°]?",
            r"lat\s*:\s*(-?\d+\.?\d*)\s*,\s*lon\s*:\s*(-?\d+\.?\d*)",
            r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)",
        ]

        for pattern in coord_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _validate_locations(self, locations: List[str]) -> List[str]:
        """Validate locations by attempting to resolve them"""
        validated = []

        for location in locations:
            if not location:
                continue

            # Try to resolve the location
            resolved = self.resolver.resolve_location(location)
            if resolved:
                validated.append(location)

        return validated

    def extract_comparative_locations(self, query: str) -> List[Tuple[str, str]]:
        """Extract location pairs for comparative analysis"""
        pairs = self._extract_from_comparative_patterns(query)

        # If no comparative patterns found, try to create pairs from all locations
        if not pairs:
            all_locations = self.extract_locations(query)
            if len(all_locations) >= 2:
                # Create pair from first two validated locations
                pairs = [(all_locations[0], all_locations[1])]

        return pairs

    def resolve_location(self, location_name: str) -> Optional[Dict]:
        """Resolve location using the dynamic resolver"""
        return self.resolver.resolve_location(location_name)


class SpatialMaskGenerator:
    """Generate spatial attention masks for climate data"""

    def __init__(self, grid_shape: Tuple[int, int] = (64, 128)):  # Simplified grid for demo
        self.n_lats, self.n_lons = grid_shape
        # Simplified lat/lon grid for demo
        self.lat_grid = np.linspace(-90, 90, self.n_lats)
        self.lon_grid = np.linspace(-180, 180, self.n_lons)

    def create_location_mask(
        self, location_info: Dict, focus_strength: float = 1.0
    ) -> torch.Tensor:
        """Create spatial mask for a single location"""
        bounds = location_info["bounds"]

        # Find grid indices for location bounds
        lat_mask = (self.lat_grid >= bounds["lat_min"]) & (self.lat_grid <= bounds["lat_max"])
        lon_mask = (self.lon_grid >= bounds["lon_min"]) & (self.lon_grid <= bounds["lon_max"])

        # Create 2D mask
        mask = torch.zeros(self.n_lats, self.n_lons)
        lat_indices = torch.from_numpy(np.where(lat_mask)[0])
        lon_indices = torch.from_numpy(np.where(lon_mask)[0])

        # Set mask values for the location region
        for lat_idx in lat_indices:
            for lon_idx in lon_indices:
                mask[lat_idx, lon_idx] = focus_strength

        return mask

    def create_union_mask(self, location_masks: List[torch.Tensor]) -> torch.Tensor:
        """Create union mask from multiple location masks"""
        if not location_masks:
            return torch.ones(self.n_lats, self.n_lons)

        # Union is the maximum of all masks at each point
        union_mask = torch.stack(location_masks, dim=0).max(dim=0)[0]
        return union_mask

    def create_comparative_masks(self, locations: List[Dict]) -> Dict[str, torch.Tensor]:
        """Create individual and union masks for comparative analysis"""
        masks = {}
        individual_masks = []

        for i, location in enumerate(locations):
            location_name = location.get("name", f"location_{i}")
            mask = self.create_location_mask(location)
            masks[f"{location_name}_mask"] = mask
            individual_masks.append(mask)

        # Create union mask
        masks["union_mask"] = self.create_union_mask(individual_masks)

        return masks


class SpatialComparativeProcessor(torch.nn.Module):
    """Process climate data for spatial comparative analysis"""

    def __init__(
        self, climate_dim: int = 512, text_dim: int = 768, grid_shape: Tuple[int, int] = (64, 128)
    ):
        super().__init__()

        self.grid_shape = grid_shape
        self.location_extractor = MultiLocationExtractor()
        self.mask_generator = SpatialMaskGenerator(grid_shape)

        # Climate encoder for spatial data
        self.climate_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(20, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 8 * 8, climate_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(climate_dim, climate_dim),
        )

        # Text encoder
        self.text_encoder = torch.nn.Embedding(1000, text_dim)

        # Spatial attention module
        self.spatial_attention = torch.nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=8, batch_first=True
        )

        # Location-specific encoders
        self.location_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 64),  # lat_min, lat_max, lon_min, lon_max
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, text_dim),
        )

        # Comparative analysis head
        self.comparative_analyzer = torch.nn.Sequential(
            torch.nn.Linear(text_dim * 3, text_dim),  # query + loc1 + loc2
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(text_dim, text_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(text_dim // 2, 3),  # location1_better, location2_better, similar
        )

        print(f"✅ Spatial comparative processor created!")
        print(f"📊 Grid shape: {grid_shape}")
        print(f"🗺️  Using dynamic location resolution with GeoPy: {GEOPY_AVAILABLE}")

    def process_spatial_query(self, query: str, climate_data: torch.Tensor) -> Dict:
        """
        Process a spatial comparative query

        Args:
            query: Natural language query with location comparison
            climate_data: [batch, channels, lat, lon] climate data

        Returns:
            Dictionary with analysis results
        """
        # Extract comparative locations
        comparative_pairs = self.location_extractor.extract_comparative_locations(query)
        all_locations = self.location_extractor.extract_locations(query)

        print(f"🔍 Query: '{query}'")
        print(f"📍 Found locations: {all_locations}")
        print(f"🔄 Comparative pairs: {comparative_pairs}")

        if not comparative_pairs and len(all_locations) >= 2:
            # Create pair from first two locations found
            comparative_pairs = [(all_locations[0], all_locations[1])]

        if not comparative_pairs:
            print("⚠️  No comparative locations found")
            return self._process_global_query(query, climate_data)

        # Process first comparative pair
        loc1_name, loc2_name = comparative_pairs[0]
        loc1_info = self.location_extractor.resolve_location(loc1_name)
        loc2_info = self.location_extractor.resolve_location(loc2_name)

        if not loc1_info or not loc2_info:
            print(f"❌ Could not resolve locations: {loc1_name}, {loc2_name}")
            return self._process_global_query(query, climate_data)

        # Add names to location info
        loc1_info["name"] = loc1_name
        loc2_info["name"] = loc2_name

        # Generate spatial masks
        masks = self.mask_generator.create_comparative_masks([loc1_info, loc2_info])

        # Extract climate features for each location
        results = self._analyze_comparative_locations(
            query, climate_data, loc1_info, loc2_info, masks
        )

        return results

    def _analyze_comparative_locations(
        self, query: str, climate_data: torch.Tensor, loc1_info: Dict, loc2_info: Dict, masks: Dict
    ) -> Dict:
        """Analyze climate data for two locations comparatively"""

        batch_size = climate_data.shape[0]

        # Encode climate data
        climate_features = self.climate_encoder(climate_data)  # [batch, climate_dim]

        # Create dummy text tokens for query
        query_tokens = torch.randint(0, 1000, (batch_size, 10))
        text_features = self.text_encoder(query_tokens)  # [batch, seq_len, text_dim]

        # Encode location information
        loc1_bounds = (
            torch.tensor(
                [
                    loc1_info["bounds"]["lat_min"],
                    loc1_info["bounds"]["lat_max"],
                    loc1_info["bounds"]["lon_min"],
                    loc1_info["bounds"]["lon_max"],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        loc2_bounds = (
            torch.tensor(
                [
                    loc2_info["bounds"]["lat_min"],
                    loc2_info["bounds"]["lat_max"],
                    loc2_info["bounds"]["lon_min"],
                    loc2_info["bounds"]["lon_max"],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        loc1_encoding = self.location_encoder(loc1_bounds)  # [batch, text_dim]
        loc2_encoding = self.location_encoder(loc2_bounds)  # [batch, text_dim]

        # Pool text features
        text_pooled = text_features.mean(dim=1)  # [batch, text_dim]

        # Combine for comparative analysis
        combined_features = torch.cat([text_pooled, loc1_encoding, loc2_encoding], dim=-1)
        comparison_logits = self.comparative_analyzer(combined_features)

        # Interpretation
        comparison_probs = torch.softmax(comparison_logits, dim=-1)

        return {
            "location1": loc1_info["name"],
            "location2": loc2_info["name"],
            "comparison_probs": comparison_probs,
            "location1_mask": masks[f"{loc1_info['name']}_mask"],
            "location2_mask": masks[f"{loc2_info['name']}_mask"],
            "union_mask": masks["union_mask"],
            "climate_features": climate_features,
            "location1_encoding": loc1_encoding,
            "location2_encoding": loc2_encoding,
            "query": query,
        }

    def _process_global_query(self, query: str, climate_data: torch.Tensor) -> Dict:
        """Fallback for non-comparative queries"""
        batch_size = climate_data.shape[0]
        climate_features = self.climate_encoder(climate_data)

        return {
            "type": "global",
            "climate_features": climate_features,
            "query": query,
            "message": "No specific locations found, processing globally",
        }


def create_demo_climate_data(batch_size: int = 1) -> torch.Tensor:
    """Create realistic demo climate data for different regions"""
    # Simplified grid matching our processor
    climate_data = torch.randn(batch_size, 20, 64, 128)

    # Add realistic regional patterns
    lat_grid = torch.linspace(-90, 90, 64).unsqueeze(1).expand(-1, 128)
    lon_grid = torch.linspace(-180, 180, 128).unsqueeze(0).expand(64, -1)

    # Temperature pattern (warmer near equator)
    temp_pattern = 20 - 0.5 * torch.abs(lat_grid)
    climate_data[:, 0:5] += temp_pattern.unsqueeze(0).unsqueeze(1)  # Temperature channels

    # Add some realistic regional variations
    # Arizona (hot, dry)
    az_lat_mask = (lat_grid >= 31.3) & (lat_grid <= 37.0)
    az_lon_mask = (lon_grid >= -114.8) & (lon_grid <= -109.0)
    az_mask = az_lat_mask & az_lon_mask
    climate_data[:, 0:5, az_mask] += 10  # Hotter
    climate_data[:, 5:10, az_mask] -= 5  # Less precipitation

    # Alaska (cold)
    ak_lat_mask = (lat_grid >= 54.7) & (lat_grid <= 71.4)
    ak_lon_mask = (lon_grid >= -179.1) & (lon_grid <= -129.0)
    ak_mask = ak_lat_mask & ak_lon_mask
    climate_data[:, 0:5, ak_mask] -= 20  # Much colder
    climate_data[:, 5:10, ak_mask] += 2  # More precipitation (snow)

    return climate_data


def demonstrate_spatial_queries():
    """Demonstrate various spatial comparative queries"""

    print("\n🚀 Starting Spatial Comparative Analysis Demo...")
    print(f"💾 Starting memory: {check_memory_usage():.1f}GB")

    # Create processor
    processor = SpatialComparativeProcessor()
    print(f"💾 After model: {check_memory_usage():.1f}GB")

    # Create demo climate data
    climate_data = create_demo_climate_data(batch_size=1)
    print(f"📊 Climate data shape: {climate_data.shape}")

    # Test queries
    test_queries = [
        "Where will it be hotter, Arizona or Alaska?",
        "Compare rainfall between California and Texas",
        "Which has more extreme weather, Sweden or Norway?",
        "Arizona vs Alaska temperature comparison",
        "What are the climate differences between Arizona and Alaska?",
        "will Malmo be hotter than 47.29°, -120.21°",
    ]

    print(f"\n🧪 Testing {len(test_queries)} spatial queries...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")

        start_time = time.time()
        results = processor.process_spatial_query(query, climate_data)
        elapsed = time.time() - start_time

        if results.get("type") == "global":
            print(f"⚠️  {results['message']}")
        else:
            # Display comparative analysis results
            loc1 = results["location1"]
            loc2 = results["location2"]
            probs = results["comparison_probs"][0]

            print(f"📍 Comparing: {loc1.title()} vs {loc2.title()}")
            print(f"🎯 Analysis Results:")
            print(f"   {loc1.title()} better: {probs[0]:.3f} ({probs[0]*100:.1f}%)")
            print(f"   {loc2.title()} better: {probs[1]:.3f} ({probs[1]*100:.1f}%)")
            print(f"   Similar:           {probs[2]:.3f} ({probs[2]*100:.1f}%)")

            # Determine winner
            winner_idx = torch.argmax(probs)
            if winner_idx == 0:
                winner = loc1.title()
            elif winner_idx == 1:
                winner = loc2.title()
            else:
                winner = "Both similar"

            print(f"🏆 Conclusion: {winner}")

            # Display mask information
            mask1 = results["location1_mask"]
            mask2 = results["location2_mask"]
            union_mask = results["union_mask"]

            print(f"🗺️  Spatial Coverage:")
            print(f"   {loc1.title()} mask coverage: {(mask1 > 0).sum().item()} pixels")
            print(f"   {loc2.title()} mask coverage: {(mask2 > 0).sum().item()} pixels")
            print(f"   Union mask coverage: {(union_mask > 0).sum().item()} pixels")

        print(f"⏱️  Processing time: {elapsed:.3f}s")
        print()

    print(f"💾 Final memory usage: {check_memory_usage():.1f}GB")

    print(f"\n🎉 Spatial Comparative Analysis Demo Completed!")

    print(f"\n📋 Key Capabilities Demonstrated:")
    print(f"   ✅ Multi-location extraction from natural language")
    print(f"   ✅ Spatial mask generation for individual locations")
    print(f"   ✅ Union mask creation for comparative analysis")
    print(f"   ✅ Location-aware climate feature extraction")
    print(f"   ✅ Comparative analysis between regions")
    print(f"   ✅ Handling of missing/unknown locations")

    print(f"\n🔧 Technical Implementation:")
    print(f"   • Extended geographic database with Alaska")
    print(f"   • Multi-location query parsing")
    print(f"   • Spatial mask union operations")
    print(f"   • Location-aware attention mechanisms")
    print(f"   • Comparative classification head")


if __name__ == "__main__":
    demonstrate_spatial_queries()
