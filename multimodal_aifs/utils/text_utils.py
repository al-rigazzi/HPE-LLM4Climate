"""
Text Processing Utilities

This module provides utilities for processing text data for AIFS multimodal analysis,
including climate-specific text preprocessing, tokenization, and embedding preparation.
"""

import re
from typing import Any

import numpy as np
import torch

# Climate-specific vocabularies and terms
CLIMATE_KEYWORDS = {
    # Weather phenomena
    "weather": [
        "rain",
        "snow",
        "storm",
        "hurricane",
        "typhoon",
        "cyclone",
        "tornado",
        "drought",
        "flood",
        "heatwave",
        "cold snap",
        "blizzard",
        "thunderstorm",
        "lightning",
        "hail",
        "fog",
        "mist",
        "cloud",
        "sunshine",
        "wind",
        "winds",
        "storms",
        "hurricanes",
        "rains",
        "floods",
        "droughts",
        "clouds",
        "temperature",
        "hot",
        "cold",
        "warm",
        "cool",
    ],
    # Climate variables
    "variables": [
        "temperature",
        "pressure",
        "humidity",
        "precipitation",
        "wind speed",
        "visibility",
        "dewpoint",
        "heat index",
        "wind chill",
        "uv index",
    ],
    # Locations
    "locations": [
        "arctic",
        "tropical",
        "temperate",
        "desert",
        "coastal",
        "inland",
        "mountain",
        "valley",
        "ocean",
        "continent",
        "island",
        "polar",
    ],
    # Time references
    "temporal": [
        "daily",
        "weekly",
        "monthly",
        "seasonal",
        "annual",
        "decadal",
        "morning",
        "afternoon",
        "evening",
        "night",
        "spring",
        "summer",
        "autumn",
        "winter",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ],
    # Measurements
    "units": [
        "celsius",
        "fahrenheit",
        "kelvin",
        "pascal",
        "millibar",
        "hpa",
        "meters",
        "kilometers",
        "miles",
        "knots",
        "mph",
        "kph",
        "mmHg",
    ],
    # Trends and patterns
    "trends": [
        "increasing",
        "decreasing",
        "stable",
        "rising",
        "falling",
        "variable",
        "anomaly",
        "normal",
        "above average",
        "below average",
        "extreme",
    ],
    # Impacts
    "impacts": [
        "agriculture",
        "ecosystem",
        "wildlife",
        "human health",
        "economy",
        "infrastructure",
        "transportation",
        "energy",
        "water resources",
    ],
}

# Common climate-related phrases
CLIMATE_PHRASES = [
    "climate change",
    "global warming",
    "greenhouse effect",
    "carbon emissions",
    "sea level rise",
    "ice cap melting",
    "ozone depletion",
    "acid rain",
    "el nino",
    "la nina",
    "jet stream",
    "pressure system",
]

# Geographic location patterns
LOCATION_PATTERNS = [
    r"\b(?:in|at|near|around|over|across)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:area|region|coast|valley|mountains?)\b",
    r"\b(?:north|south|east|west)(?:ern)?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
]


def extract_location_keywords(text: str) -> list[str]:
    """
    Extract location-related keywords from text.

    Args:
        text: Input text to analyze

    Returns:
        list of extracted location keywords
    """
    locations = []
    text_lower = text.lower()

    # Check for known location terms
    known_locations = [
        "new york",
        "london",
        "paris",
        "tokyo",
        "sydney",
        "san francisco",
        "miami",
        "chicago",
        "los angeles",
        "beijing",
        "northern california",
        "southern california",
        "east coast",
        "west coast",
        "midwest",
        "great lakes",
        "pacific",
        "atlantic",
        "gulf",
        "caribbean",
    ]

    for location in known_locations:
        if location in text_lower:
            locations.append(location)

    # Use regex patterns to find location references
    for pattern in LOCATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_locations = []
    for loc in locations:
        if loc.lower() not in seen:
            seen.add(loc.lower())
            unique_locations.append(loc)

    return unique_locations


def parse_climate_query(text: str) -> dict[str, list[str]]:
    """
    Parse a climate query to extract key components.

    Args:
        text: Climate query text

    Returns:
        Dictionary with parsed components
    """
    text_lower = text.lower()

    result: dict[str, list[str]] = {
        "variables": [],
        "locations": [],
        "temporal": [],
        "phenomena": [],
        "trends": [],
    }

    # Extract climate variables
    for var in CLIMATE_KEYWORDS["variables"]:
        if var in text_lower:
            result["variables"].append(var)

    # Extract locations
    result["locations"] = extract_location_keywords(text)

    # Extract temporal references
    for temp in CLIMATE_KEYWORDS["temporal"]:
        if temp in text_lower:
            result["temporal"].append(temp)

    # Extract weather phenomena
    for weather in CLIMATE_KEYWORDS["weather"]:
        if weather in text_lower:
            result["phenomena"].append(weather)

    # Extract trends
    for trend in CLIMATE_KEYWORDS["trends"]:
        if trend in text_lower:
            result["trends"].append(trend)

    return result


# Location patterns for structured parsing
LOCATION_PARSE_PATTERNS = {
    "coordinates": r"(-?\d+\.?\d*)\s*[°,]\s*(-?\d+\.?\d*)",
    "city_country": r"([A-Za-z\s]+),\s*([A-Za-z\s]+)",
    "region": r"(north|south|east|west|central|upper|lower)\s+([A-Za-z\s]+)",
    "compass": r"(north|south|east|west|northeast|northwest|southeast|southwest)",
}


class ClimateTextProcessor:
    """
    Processor for climate-related text data.

    This class provides methods for preprocessing, tokenizing, and extracting
    information from climate-related text descriptions.
    """

    def __init__(
        self,
        max_length: int = 512,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        expand_contractions: bool = True,
    ):
        """
        Initialize climate text processor.

        Args:
            max_length: Maximum text length
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            expand_contractions: Whether to expand contractions
        """
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.expand_contractions = expand_contractions

        # Build comprehensive vocabulary
        self.climate_vocab = set()
        for _, words in CLIMATE_KEYWORDS.items():
            self.climate_vocab.update(words)

        # Add phrases
        for phrase in CLIMATE_PHRASES:
            self.climate_vocab.update(phrase.split())

    def preprocess_text(self, text: str | None) -> str:
        """
        Preprocess text for climate analysis.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """

        # Handle None input
        if text is None:
            return ""

        # Basic cleaning
        text = text.strip()

        # Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()

        # Expand contractions if requested
        if self.expand_contractions:
            text = self._expand_contractions(text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Handle special characters
        text = text.replace("\n", " ").replace("\t", " ")

        # Remove punctuation if requested
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text)

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            "it's": "it is",
            "isn't": "is not",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "mustn't": "must not",
            "aren't": "are not",
            "weren't": "were not",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "where's": "where is",
            "how's": "how is",
            "that's": "that is",
        }

        for contraction, expansion in contractions.items():
            text = re.sub(
                r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.IGNORECASE
            )

        return text

    def extract_climate_keywords(self, text: str) -> list[str]:
        """
        Extract climate-related keywords from text.

        Args:
            text: Input text

        Returns:
            list of found climate keywords
        """
        text = self.preprocess_text(text)
        words = text.split()

        found_keywords = []
        for word in words:
            if word.lower() in self.climate_vocab:
                found_keywords.append(word.lower())

        # Also check for phrases
        for phrase in CLIMATE_PHRASES:
            if phrase.lower() in text.lower():
                found_keywords.append(phrase.lower())

        return list(set(found_keywords))  # Remove duplicates

    def extract_locations(self, text: str) -> list[dict[str, Any]]:
        """
        Extract location references from text.

        Args:
            text: Input text

        Returns:
            List of location dictionaries
        """
        locations = []

        # Extract coordinates
        coord_matches = re.finditer(LOCATION_PARSE_PATTERNS["coordinates"], text)
        for match in coord_matches:
            lat, lon = match.groups()
            locations.append(
                {
                    "type": "coordinates",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "text": match.group(),
                }
            )

        # Extract city, country patterns
        city_matches = re.finditer(LOCATION_PARSE_PATTERNS["city_country"], text)
        for match in city_matches:
            city, country = match.groups()
            locations.append(
                {
                    "type": "city_country",
                    "city": city.strip(),
                    "country": country.strip(),
                    "text": match.group(),
                }
            )

        # Extract regional descriptions
        region_matches = re.finditer(LOCATION_PARSE_PATTERNS["region"], text, re.IGNORECASE)
        for match in region_matches:
            direction, region = match.groups()
            locations.append(
                {
                    "type": "regional",
                    "direction": direction.lower(),
                    "region": region.strip(),
                    "text": match.group(),
                }
            )

        return locations

    def extract_numerical_values(self, text: str) -> list[dict[str, str | float]]:
        """
        Extract numerical values and their units from text.

        Args:
            text: Input text

        Returns:
            list of numerical value dictionaries
        """
        values = []

        # Pattern for numbers with units
        number_unit_pattern = r"(-?\d+\.?\d*)\s*([A-Za-z°%]+)"
        matches = re.finditer(number_unit_pattern, text)

        for match in matches:
            value_str, unit = match.groups()
            try:
                value = float(value_str)
                values.append(
                    {
                        "value": value,
                        "unit": unit.lower(),
                        "text": match.group(),
                        "position": match.start(),
                    }
                )
            except ValueError:
                continue

        return values

    def categorize_text(self, text: str) -> dict[str, float]:
        """
        Categorize text by climate-related themes.

        Args:
            text: Input text

        Returns:
            Dictionary of category scores
        """
        text = self.preprocess_text(text)
        words = set(text.split())

        category_scores = {}
        total_words = len(words)

        if total_words == 0:
            return {category: 0.0 for category in CLIMATE_KEYWORDS}

        for category, keywords in CLIMATE_KEYWORDS.items():
            matching_words = words.intersection(set(keywords))
            score = len(matching_words) / total_words
            category_scores[category] = score

        return category_scores

    def create_text_features(
        self, text: str
    ) -> dict[str, int | float | list[Any] | dict[str, float]]:
        """
        Create comprehensive features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of text features
        """
        original_text = text
        processed_text = self.preprocess_text(text)

        # Basic features
        features: dict[str, int | float | list[Any] | dict[str, float]] = {
            "original_length": len(original_text),
            "processed_length": len(processed_text),
            "length": len(original_text),  # Alias for compatibility
            "word_count": len(processed_text.split()),
            "character_count": len(processed_text),
        }

        # Climate-specific features
        climate_keywords = self.extract_climate_keywords(text)
        features["climate_keywords"] = climate_keywords
        features["climate_keyword_count"] = len(climate_keywords)

        # Location features
        locations = self.extract_locations(text)
        features["locations"] = locations
        features["location_count"] = len(locations)

        # Numerical features
        numerical_values = self.extract_numerical_values(text)
        features["numerical_values"] = numerical_values
        features["numerical_count"] = len(numerical_values)

        # Category scores
        category_scores = self.categorize_text(text)
        features["category_scores"] = category_scores

        # Overall climate relevance score
        total_climate_score = sum(category_scores.values())
        features["climate_relevance"] = min(total_climate_score, 1.0)

        return features


class TextEmbeddingUtils:
    """
    Utilities for creating text embeddings for multimodal analysis.
    """

    def __init__(self, embedding_dim: int = 768, vocab_size: int = 10000):
        """
        Initialize text embedding utilities.

        Args:
            embedding_dim: Dimension of embeddings
            vocab_size: Vocabulary size
        """
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.embeddings: torch.Tensor | None = None
        self.is_fitted = False

    def build_vocabulary(self, texts: list[str]) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: list of texts to build vocabulary from
        """
        word_counts: dict[str, int] = {}

        # Count words
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Build mappings
        self.word_to_idx = {"<UNK>": 0, "<PAD>": 1}
        self.idx_to_word = {0: "<UNK>", 1: "<PAD>"}

        for i, (word, _) in enumerate(sorted_words[: self.vocab_size - 2]):
            idx = i + 2
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        # Initialize embeddings
        self.embeddings = torch.randn(len(self.word_to_idx), self.embedding_dim)
        self.is_fitted = True

    def text_to_indices(self, text: str, max_length: int | None = None) -> list[int]:
        """
        Convert text to indices.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            list of indices
        """
        if not self.is_fitted:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary first.")

        words = text.lower().split()
        indices = [self.word_to_idx.get(word, 0) for word in words]  # 0 is <UNK>

        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                # Pad with <PAD> token (index 1)
                indices.extend([1] * (max_length - len(indices)))

        return indices

    def create_positional_encoding(self, seq_length: int) -> torch.Tensor:
        """
        Create positional encoding for sequence.

        Args:
            seq_length: Length of sequence

        Returns:
            Positional encoding tensor
        """
        position = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2).float() * -(np.log(10000.0) / self.embedding_dim)
        )

        pos_encoding = torch.zeros(seq_length, self.embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def embed_text(
        self,
        text: str,
        max_length: int = 128,
        include_positional: bool = True,
        use_positional: bool | None = None,
    ) -> torch.Tensor:
        """
        Create embeddings for text.

        Args:
            text: Input text
            max_length: Maximum sequence length
            include_positional: Whether to add positional encoding
            use_positional: Alternative name for include_positional (for compatibility)

        Returns:
            Text embeddings tensor
        """
        if not self.is_fitted:
            raise RuntimeError("Vocabulary not built. Call build_vocabulary first.")

        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized")

        # Handle both parameter names for compatibility
        if use_positional is not None:
            include_positional = use_positional

        indices = self.text_to_indices(text, max_length)
        embeddings = self.embeddings[indices]

        if include_positional:
            pos_encoding = self.create_positional_encoding(len(indices))
            embeddings += pos_encoding

        return embeddings


def test_text_processing():
    """Test text processing utilities."""
    print("Testing Text Processing Utilities")
    print("=" * 40)

    # Test text processor
    processor = ClimateTextProcessor()

    # Sample climate text
    sample_text = """
    The temperature in London, UK reached 35°C yesterday, which is 5°C above average
    for this time of year. Strong winds from the southwest brought humid air from
    the Atlantic Ocean. Climate change is causing more frequent heatwaves across
    Western Europe. The coordinates 51.5°, -0.1° show the exact location of the
    weather station.
    """

    # Test preprocessing
    processed = processor.preprocess_text(sample_text)
    print(f"Original length: {len(sample_text)}")
    print(f"Processed length: {len(processed)}")

    # Test keyword extraction
    keywords = processor.extract_climate_keywords(sample_text)
    print(f"Climate keywords found: {keywords}")

    # Test location extraction
    locations = processor.extract_locations(sample_text)
    print(f"Locations found: {len(locations)}")
    for loc in locations:
        print(f"  {loc}")

    # Test numerical extraction
    values = processor.extract_numerical_values(sample_text)
    print(f"Numerical values: {len(values)}")
    for val in values:
        print(f"  {val}")

    # Test categorization
    categories = processor.categorize_text(sample_text)
    print(f"Category scores: {categories}")

    # Test comprehensive features
    features = processor.create_text_features(sample_text)
    print(f"Climate relevance score: {features['climate_relevance']:.3f}")

    # Test embedding utilities
    texts = [sample_text, "Normal weather conditions", "Heavy rainfall expected"]

    embedder = TextEmbeddingUtils(embedding_dim=128, vocab_size=1000)
    embedder.build_vocabulary(texts)

    print(f"Vocabulary size: {len(embedder.word_to_idx)}")

    # Test text embedding
    embedding = embedder.embed_text(sample_text, max_length=64)
    print(f"Text embedding shape: {embedding.shape}")

    print("All text processing tests passed!")


if __name__ == "__main__":
    test_text_processing()
