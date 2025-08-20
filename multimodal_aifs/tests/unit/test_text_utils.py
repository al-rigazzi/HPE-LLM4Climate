#!/usr/bin/env python3
"""
Unit tests for text processing utilities.

This module tests the text processing utilities for AIFS multimodal analysis,
including climate-specific text preprocessing, tokenization, and embedding preparation.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import pytest

from multimodal_aifs.utils.text_utils import (
    extract_location_keywords,
    parse_climate_query,
    ClimateTextProcessor,
    TextEmbeddingUtils,
    CLIMATE_KEYWORDS,
)


class TestLocationExtraction(unittest.TestCase):
    """Test cases for location extraction functions."""

    def test_extract_location_keywords_basic(self):
        """Test basic location keyword extraction."""
        text = "The weather in New York is sunny today."
        locations = extract_location_keywords(text)

        self.assertIsInstance(locations, list)
        # Should find "New York" as a location
        self.assertTrue(any("new york" in loc.lower() for loc in locations))

    def test_extract_location_keywords_multiple(self):
        """Test extraction of multiple location keywords."""
        text = "Compare weather between London, Paris, and Tokyo."
        locations = extract_location_keywords(text)

        self.assertIsInstance(locations, list)
        self.assertGreater(len(locations), 0)

        # Should find multiple locations
        text_lower = text.lower()
        for location in ["london", "paris", "tokyo"]:
            self.assertTrue(any(location in loc.lower() for loc in locations))

    def test_extract_location_keywords_no_locations(self):
        """Test text with no location keywords."""
        text = "The temperature is rising due to climate change."
        locations = extract_location_keywords(text)

        self.assertIsInstance(locations, list)
        # May or may not find locations depending on implementation

    def test_extract_location_keywords_empty_text(self):
        """Test extraction from empty text."""
        locations = extract_location_keywords("")
        self.assertIsInstance(locations, list)
        self.assertEqual(len(locations), 0)


class TestClimateQueryParsing(unittest.TestCase):
    """Test cases for climate query parsing."""

    def test_parse_climate_query_basic(self):
        """Test basic climate query parsing."""
        text = "What is the temperature and humidity in Miami today?"
        parsed = parse_climate_query(text)

        self.assertIsInstance(parsed, dict)
        self.assertIn("variables", parsed)
        self.assertIn("locations", parsed)

        # Should identify climate variables
        variables = [var.lower() for var in parsed["variables"]]
        self.assertTrue(any("temperature" in var for var in variables))
        self.assertTrue(any("humidity" in var for var in variables))

    def test_parse_climate_query_multiple_variables(self):
        """Test parsing query with multiple climate variables."""
        text = "Show me wind speed, pressure, and precipitation data for California."
        parsed = parse_climate_query(text)

        self.assertIsInstance(parsed, dict)
        variables = [var.lower() for var in parsed["variables"]]

        expected_vars = ["wind", "pressure", "precipitation"]
        for expected in expected_vars:
            self.assertTrue(any(expected in var for var in variables))

    def test_parse_climate_query_temporal(self):
        """Test parsing query with temporal information."""
        text = "Historical temperature trends over the past decade in Europe."
        parsed = parse_climate_query(text)

        self.assertIsInstance(parsed, dict)
        # Should handle temporal keywords if implemented

    def test_parse_climate_query_empty(self):
        """Test parsing empty query."""
        parsed = parse_climate_query("")

        self.assertIsInstance(parsed, dict)
        self.assertIn("variables", parsed)
        self.assertIn("locations", parsed)


class TestClimateTextProcessor(unittest.TestCase):
    """Test cases for ClimateTextProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = ClimateTextProcessor(
            lowercase=True,
            remove_punctuation=True,
            max_length=512
        )

    def test_initialization_default(self):
        """Test ClimateTextProcessor initialization with defaults."""
        processor = ClimateTextProcessor()

        self.assertTrue(processor.lowercase)
        self.assertTrue(processor.remove_punctuation)
        self.assertEqual(processor.max_length, 512)
        self.assertTrue(processor.expand_contractions)

    def test_initialization_custom(self):
        """Test ClimateTextProcessor initialization with custom parameters."""
        processor = ClimateTextProcessor(
            lowercase=False,
            remove_punctuation=False,
            max_length=256,
            expand_contractions=False
        )

        self.assertFalse(processor.lowercase)
        self.assertFalse(processor.remove_punctuation)
        self.assertEqual(processor.max_length, 256)
        self.assertFalse(processor.expand_contractions)

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "The WEATHER in New York is AMAZING!"
        processed = self.processor.preprocess_text(text)

        self.assertIsInstance(processed, str)
        # Should be lowercase (if enabled)
        if self.processor.lowercase:
            self.assertEqual(processed, processed.lower())

        # Should remove punctuation (if enabled)
        if self.processor.remove_punctuation:
            self.assertNotIn("!", processed)

    def test_preprocess_text_contractions(self):
        """Test contraction expansion in preprocessing."""
        text = "It's raining and there isn't any sunshine."
        processor = ClimateTextProcessor(expand_contractions=True)
        processed = processor.preprocess_text(text)

        # Should expand contractions if enabled
        self.assertNotIn("isn't", processed)
        self.assertNotIn("it's", processed.lower())

    def test_preprocess_text_empty(self):
        """Test preprocessing empty text."""
        processed = self.processor.preprocess_text("")
        self.assertEqual(processed, "")

    def test_extract_climate_keywords(self):
        """Test climate keyword extraction."""
        text = "The hurricane brought heavy rain and strong winds."
        keywords = self.processor.extract_climate_keywords(text)

        self.assertIsInstance(keywords, list)

        # Should find climate-related keywords
        keywords_lower = [kw.lower() for kw in keywords]
        expected_keywords = ["hurricane", "rain", "wind"]
        for expected in expected_keywords:
            self.assertTrue(any(expected in kw for kw in keywords_lower))

    def test_extract_locations(self):
        """Test location extraction."""
        text = "Weather forecast for Los Angeles and San Francisco."
        locations = self.processor.extract_locations(text)

        self.assertIsInstance(locations, list)
        if len(locations) > 0:
            self.assertIsInstance(locations[0], dict)

    def test_extract_numerical_values(self):
        """Test numerical value extraction."""
        text = "Temperature reached 75.5°F with humidity at 60% today."
        values = self.processor.extract_numerical_values(text)

        self.assertIsInstance(values, list)

        # Should find numerical values
        if len(values) > 0:
            self.assertIsInstance(values[0], dict)
            # Should contain value and possibly unit information

    def test_categorize_text(self):
        """Test text categorization."""
        weather_text = "Today will be sunny with light winds."
        forecast_text = "Tomorrow's forecast shows rain likely."
        alert_text = "Hurricane warning issued for coastal areas."

        weather_score = self.processor.categorize_text(weather_text)
        forecast_score = self.processor.categorize_text(forecast_text)
        alert_score = self.processor.categorize_text(alert_text)

        self.assertIsInstance(weather_score, dict)
        self.assertIsInstance(forecast_score, dict)
        self.assertIsInstance(alert_score, dict)

        # Check that scores are between 0 and 1
        for scores in [weather_score, forecast_score, alert_score]:
            for category, score in scores.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_create_text_features(self):
        """Test text feature creation."""
        text = "Severe thunderstorm warning for downtown Chicago area."
        features = self.processor.create_text_features(text)

        self.assertIsInstance(features, dict)

        # Should contain various feature types
        expected_keys = ["length", "word_count", "climate_keywords", "locations"]
        for key in expected_keys:
            self.assertIn(key, features)

        # Check feature types
        self.assertIsInstance(features["length"], int)
        self.assertIsInstance(features["word_count"], int)
        self.assertIsInstance(features["climate_keywords"], list)


class TestTextEmbeddingUtils(unittest.TestCase):
    """Test cases for TextEmbeddingUtils class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.embedding_utils = TextEmbeddingUtils(embedding_dim=128, vocab_size=1000)

    def test_initialization(self):
        """Test TextEmbeddingUtils initialization."""
        self.assertEqual(self.embedding_utils.embedding_dim, 128)
        self.assertEqual(self.embedding_utils.vocab_size, 1000)
        self.assertIsInstance(self.embedding_utils.word_to_idx, dict)
        self.assertIsInstance(self.embedding_utils.idx_to_word, dict)

    def test_build_vocabulary(self):
        """Test vocabulary building."""
        texts = [
            "The weather is sunny today",
            "Rain is expected tomorrow",
            "Hurricane watch issued for coast"
        ]

        self.embedding_utils.build_vocabulary(texts)

        # Check that vocabulary was built
        self.assertGreater(len(self.embedding_utils.word_to_idx), 0)
        self.assertGreater(len(self.embedding_utils.idx_to_word), 0)

        # Check special tokens
        self.assertIn("<PAD>", self.embedding_utils.word_to_idx)
        self.assertIn("<UNK>", self.embedding_utils.word_to_idx)

        # Check that common words are in vocabulary
        common_words = ["weather", "rain", "today"]
        for word in common_words:
            if word in " ".join(texts).lower():
                self.assertIn(word, self.embedding_utils.word_to_idx)

    def test_text_to_indices(self):
        """Test text to indices conversion."""
        texts = ["weather is nice", "rain expected"]
        self.embedding_utils.build_vocabulary(texts)

        text = "weather is good"
        indices = self.embedding_utils.text_to_indices(text)

        self.assertIsInstance(indices, list)
        self.assertGreater(len(indices), 0)

        # All indices should be valid
        for idx in indices:
            self.assertIsInstance(idx, int)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(self.embedding_utils.word_to_idx))

    def test_text_to_indices_with_max_length(self):
        """Test text to indices conversion with max length."""
        texts = ["short text", "longer text sequence"]
        self.embedding_utils.build_vocabulary(texts)

        text = "this is a longer text sequence for testing"
        max_length = 5
        indices = self.embedding_utils.text_to_indices(text, max_length=max_length)

        self.assertLessEqual(len(indices), max_length)

    def test_create_positional_encoding(self):
        """Test positional encoding creation."""
        seq_length = 10
        pos_encoding = self.embedding_utils.create_positional_encoding(seq_length)

        self.assertIsInstance(pos_encoding, torch.Tensor)
        self.assertEqual(pos_encoding.shape, (seq_length, self.embedding_utils.embedding_dim))

    def test_embed_text(self):
        """Test text embedding."""
        texts = ["sunny weather", "rainy day", "storm approaching"]
        self.embedding_utils.build_vocabulary(texts)

        text = "sunny day"
        embedding = self.embedding_utils.embed_text(text)

        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape[1], self.embedding_utils.embedding_dim)

    def test_embed_text_with_positional(self):
        """Test text embedding with positional encoding."""
        texts = ["weather forecast", "climate data"]
        self.embedding_utils.build_vocabulary(texts)

        text = "weather"
        embedding = self.embedding_utils.embed_text(text, use_positional=True)

        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape[1], self.embedding_utils.embedding_dim)


class TestClimateKeywords(unittest.TestCase):
    """Test cases for climate keywords constants."""

    def test_climate_keywords_structure(self):
        """Test CLIMATE_KEYWORDS structure."""
        self.assertIsInstance(CLIMATE_KEYWORDS, dict)
        self.assertGreater(len(CLIMATE_KEYWORDS), 0)

        # Check that each category contains a list of keywords
        for category, keywords in CLIMATE_KEYWORDS.items():
            self.assertIsInstance(category, str)
            self.assertIsInstance(keywords, list)
            self.assertGreater(len(keywords), 0)

            # Each keyword should be a string
            for keyword in keywords:
                self.assertIsInstance(keyword, str)
                self.assertGreater(len(keyword), 0)

    def test_climate_keywords_content(self):
        """Test that CLIMATE_KEYWORDS contains expected categories."""
        expected_categories = ["weather", "variables"]

        for category in expected_categories:
            self.assertIn(category, CLIMATE_KEYWORDS)

        # Check weather keywords
        weather_keywords = CLIMATE_KEYWORDS.get("weather", [])
        expected_weather = ["rain", "snow", "storm", "wind", "temperature"]

        for expected in expected_weather:
            self.assertTrue(any(expected in keyword.lower() for keyword in weather_keywords))


class TestTextUtilsIntegration(unittest.TestCase):
    """Integration tests for text utilities."""

    def test_full_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        # Initialize processor and embedding utils
        processor = ClimateTextProcessor()
        embedding_utils = TextEmbeddingUtils(embedding_dim=64, vocab_size=500)

        # Sample climate texts
        texts = [
            "Hurricane warning issued for Miami area with winds up to 120 mph",
            "Temperature forecast shows 85°F in Los Angeles tomorrow",
            "Heavy rainfall expected in Seattle with 2 inches predicted",
            "Drought conditions continue in Phoenix with no rain for 45 days"
        ]

        # Build vocabulary
        embedding_utils.build_vocabulary(texts)

        # Process each text
        for text in texts:
            # Preprocess
            processed = processor.preprocess_text(text)
            self.assertIsInstance(processed, str)

            # Extract features
            features = processor.create_text_features(text)
            self.assertIsInstance(features, dict)

            # Create embeddings
            embedding = embedding_utils.embed_text(processed)
            self.assertIsInstance(embedding, torch.Tensor)

            # Parse climate query
            parsed = parse_climate_query(text)
            self.assertIsInstance(parsed, dict)

    def test_climate_specific_functionality(self):
        """Test climate-specific text processing functionality."""
        processor = ClimateTextProcessor()

        climate_text = "Severe thunderstorm with 70mph winds and 3-inch hail in Dallas"

        # Extract climate keywords
        keywords = processor.extract_climate_keywords(climate_text)
        self.assertIsInstance(keywords, list)
        self.assertTrue(any("storm" in kw.lower() for kw in keywords))

        # Extract numerical values (temperature, wind speed, etc.)
        values = processor.extract_numerical_values(climate_text)
        self.assertIsInstance(values, list)

        # Extract locations
        locations = processor.extract_locations(climate_text)
        self.assertIsInstance(locations, list)

        # Categorize text
        categories = processor.categorize_text(climate_text)
        self.assertIsInstance(categories, dict)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        processor = ClimateTextProcessor()
        embedding_utils = TextEmbeddingUtils()

        # Empty text
        empty_result = processor.preprocess_text("")
        self.assertEqual(empty_result, "")

        # Very long text
        long_text = "weather " * 1000
        long_result = processor.preprocess_text(long_text)
        self.assertIsInstance(long_result, str)

        # Text with special characters
        special_text = "Temperature: 72°F (22°C) with 45% humidity @#$%"
        special_result = processor.preprocess_text(special_text)
        self.assertIsInstance(special_result, str)

        # Build vocabulary with minimal data
        embedding_utils.build_vocabulary(["a", "b", "c"])
        self.assertGreater(len(embedding_utils.word_to_idx), 0)


if __name__ == "__main__":
    unittest.main()
