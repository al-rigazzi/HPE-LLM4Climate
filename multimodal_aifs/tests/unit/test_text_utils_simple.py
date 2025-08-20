#!/usr/bin/env python3
"""
Simple working tests for text utilities.
"""

import unittest
from multimodal_aifs.utils.text_utils import ClimateTextProcessor


class TestTextUtils(unittest.TestCase):
    """Test cases for text utilities."""

    def test_processor_init(self):
        """Test ClimateTextProcessor initialization."""
        processor = ClimateTextProcessor()
        self.assertIsInstance(processor, ClimateTextProcessor)

    def test_extract_locations_basic(self):
        """Test basic location extraction."""
        processor = ClimateTextProcessor()
        text = "Weather in New York is cloudy"
        locations = processor.extract_locations(text)
        self.assertIsInstance(locations, list)

    def test_extract_climate_keywords(self):
        """Test climate keyword extraction."""
        processor = ClimateTextProcessor()
        query = "What is the temperature today?"
        keywords = processor.extract_climate_keywords(query)
        self.assertIsInstance(keywords, list)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        processor = ClimateTextProcessor()
        text = "Climate change affects weather patterns"
        processed = processor.preprocess_text(text)
        self.assertIsInstance(processed, str)

    def test_process_text_empty(self):
        """Test processing empty text."""
        processor = ClimateTextProcessor()
        result = processor.preprocess_text("")
        self.assertIsInstance(result, str)

    def test_process_text_none(self):
        """Test processing None input."""
        processor = ClimateTextProcessor()
        result = processor.preprocess_text(None)
        self.assertIsInstance(result, (str, type(None)))

    def test_categorize_text(self):
        """Test text categorization."""
        processor = ClimateTextProcessor()
        query = "What's the weather like?"
        result = processor.categorize_text(query)
        self.assertIsInstance(result, dict)
        # Check that it returns expected categories
        expected_keys = ['weather', 'variables', 'locations', 'temporal', 'units', 'trends', 'impacts']
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))

    def test_create_text_features(self):
        """Test text feature creation."""
        processor = ClimateTextProcessor()
        text = "Temperature in New York will increase by 2 degrees"
        features = processor.create_text_features(text)
        self.assertIsInstance(features, dict)

    def test_extract_numerical_values(self):
        """Test numerical value extraction."""
        processor = ClimateTextProcessor()
        text = "Temperature will be 25 degrees with 80% humidity"
        values = processor.extract_numerical_values(text)
        self.assertIsInstance(values, list)

    def test_climate_vocab_property(self):
        """Test climate vocabulary property."""
        processor = ClimateTextProcessor()
        vocab = processor.climate_vocab
        self.assertIsInstance(vocab, set)
        self.assertGreater(len(vocab), 0)
        # Check that some expected climate terms are in vocab
        expected_terms = ['temperature', 'precipitation', 'humidity', 'wind']
        for term in expected_terms:
            self.assertIn(term, vocab)

    def test_processor_config_properties(self):
        """Test processor configuration properties."""
        processor = ClimateTextProcessor()
        self.assertIsInstance(processor.max_length, int)
        self.assertIsInstance(processor.lowercase, bool)
        self.assertIsInstance(processor.remove_punctuation, bool)

    def test_preprocess_text_with_options(self):
        """Test text preprocessing with different options."""
        # Test with lowercase=False
        processor_no_lower = ClimateTextProcessor(lowercase=False)
        text = "TEMPERATURE in New York"
        result = processor_no_lower.preprocess_text(text)
        self.assertIsInstance(result, str)

        # Test with remove_punctuation=False
        processor_keep_punct = ClimateTextProcessor(remove_punctuation=False)
        text_punct = "What's the weather like? It's sunny!"
        result_punct = processor_keep_punct.preprocess_text(text_punct)
        self.assertIsInstance(result_punct, str)

    def test_extract_locations_with_coordinates(self):
        """Test location extraction with various formats."""
        processor = ClimateTextProcessor()

        # Test with city names
        text1 = "Weather in Paris, London, and Tokyo"
        locations1 = processor.extract_locations(text1)
        self.assertIsInstance(locations1, list)

        # Test with coordinates
        text2 = "Location at 40.7128, -74.0060"
        locations2 = processor.extract_locations(text2)
        self.assertIsInstance(locations2, list)

    def test_extract_climate_keywords_comprehensive(self):
        """Test comprehensive climate keyword extraction."""
        processor = ClimateTextProcessor()

        # Test with weather terms
        text1 = "temperature precipitation humidity wind speed"
        keywords1 = processor.extract_climate_keywords(text1)
        self.assertIsInstance(keywords1, list)

        # Test with climate change terms
        text2 = "global warming carbon emissions greenhouse gases"
        keywords2 = processor.extract_climate_keywords(text2)
        self.assertIsInstance(keywords2, list)

        # Test with no climate terms
        text3 = "the quick brown fox jumps over the lazy dog"
        keywords3 = processor.extract_climate_keywords(text3)
        self.assertIsInstance(keywords3, list)


if __name__ == "__main__":
    unittest.main()
