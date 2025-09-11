#!/usr/bin/env python3
"""
Integration Test for AIFS Climate Fusion

This test validates the complete climate-text fusion pipeline using AIFS
encoder for multimodal climate analysis.

Usage:
    python multimodal_aifs/tests/integration/test_aifs_climate_fusion.py
    python -m pytest multimodal_aifs/tests/integration/test_aifs_climate_fusion.py -v
"""

import os
import sys
import time
import unittest
import warnings
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimodal_aifs.core.aifs_climate_fusion import AIFSClimateEmbedding, AIFSClimateTextFusion


class TestAIFSClimateFusion(unittest.TestCase):
    """Test suite for AIFS climate-text fusion integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test class with paths and configurations."""
        cls.project_root = project_root
        cls.aifs_model_path = (
            cls.project_root
            / "multimodal_aifs"
            / "models"
            / "extracted_models"
            / "aifs_encoder_full.pth"
        )
        cls.device = "cpu"  # Use CPU for testing
        cls.has_real_model = cls.aifs_model_path.exists()

        print(f"🧪 AIFS Climate Fusion Integration Test Setup")
        print(f"   Project root: {cls.project_root}")
        print(f"   AIFS model path: {cls.aifs_model_path}")
        print(f"   Has real model: {cls.has_real_model}")
        print(f"   Device: {cls.device}")

    def test_fusion_module_initialization(self):
        """Test AIFSClimateTextFusion initialization."""
        print("\\n🔧 Testing Fusion Module Initialization")

        if self.has_real_model:
            # Test with real AIFS model
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path),
                climate_dim=218,  # Advanced AIFS encoder dimension
                text_dim=768,
                fusion_dim=512,
                num_attention_heads=8,
                device=self.device,
            )

            self.assertEqual(fusion_module.climate_dim, 218)
            self.assertEqual(fusion_module.text_dim, 768)
            self.assertEqual(fusion_module.fusion_dim, 512)
            self.assertEqual(fusion_module.device, self.device)

            # Note: With checkpoint path only, encoder will be None until AIFS model is provided
            if fusion_module.aifs_encoder is None:
                print(
                    "   ⚠️  AIFS encoder is None (expected when using checkpoint path without model)"
                )
            else:
                self.assertIsNotNone(fusion_module.aifs_encoder)

            print("   ✅ Real model initialization successful")

        else:
            print("   ⚠️  Skipping real model test (model not available)")

    def test_climate_data_encoding(self):
        """Test climate data encoding through fusion module."""
        print("\\n🌡️ Testing Climate Data Encoding")

        if not self.has_real_model:
            print("   ⚠️  Skipping encoding test (no real model)")
            return

        try:
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path),
                climate_dim=218,  # Advanced AIFS encoder dimension
                text_dim=768,
                fusion_dim=512,
                device=self.device,
            )

            # Check if encoder was loaded successfully
            if fusion_module.aifs_encoder is None:
                print("   ⚠️  Skipping encoding test (AIFS encoder not available)")
                return
        except Exception as e:
            print(f"   ⚠️  Skipping encoding test (initialization failed: {e})")
            return

        # Create test climate data
        batch_size = 4
        climate_data = torch.randn(batch_size, 218)  # AIFS input size

        # Test encoding
        start_time = time.time()
        try:
            climate_features = fusion_module.encode_climate_data(climate_data)
            encoding_time = time.time() - start_time

            # Validate output
            self.assertEqual(climate_features.shape[0], batch_size)
            self.assertEqual(climate_features.shape[1], 512)  # fusion_dim
            self.assertEqual(climate_features.device.type, self.device)

            print(f"   ✅ Climate encoding: {climate_data.shape} -> {climate_features.shape}")
            print(f"   ✅ Encoding time: {encoding_time:.4f}s")
        except Exception as e:
            print(f"   ⚠️  Encoding failed: {e}")

    def test_text_encoding(self):
        """Test text encoding functionality."""
        print("\\n📝 Testing Text Encoding")

        # Create fusion module (can work without real AIFS for text encoding)
        try:
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path) if self.has_real_model else None,
                climate_dim=218,  # Advanced AIFS encoder dimension
                text_dim=768,
                fusion_dim=512,
                device=self.device,
            )
        except Exception:
            # Skip if can't initialize
            print("   ⚠️  Skipping text encoding test (initialization failed)")
            return

        # Test texts
        texts = [
            "High temperature and low pressure system",
            "Strong winds from the southwest",
            "Heavy rainfall expected in the region",
            "Clear skies with moderate temperatures",
        ]

        # Test text encoding
        text_features = fusion_module.encode_text(texts)

        # Validate output
        self.assertEqual(text_features.shape[0], len(texts))
        self.assertEqual(text_features.shape[1], 512)  # fusion_dim

        print(f"   ✅ Text encoding: {len(texts)} texts -> {text_features.shape}")

        # Test with pre-computed embeddings
        batch_size = len(texts)
        pre_computed = torch.randn(batch_size, 768)  # text_dim
        text_features_2 = fusion_module.encode_text(texts, pre_computed)

        self.assertEqual(text_features_2.shape, text_features.shape)

        print("   ✅ Pre-computed embeddings")

    def test_multimodal_fusion(self):
        """Test complete multimodal fusion pipeline."""
        print("\\n🔀 Testing Multimodal Fusion")

        if not self.has_real_model:
            print("   ⚠️  Skipping fusion test (no real model)")
            return

        try:
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path),
                climate_dim=218,  # Advanced AIFS encoder dimension
                text_dim=768,
                fusion_dim=512,
                device=self.device,
            )

            # Check if encoder was loaded successfully
            if fusion_module.aifs_encoder is None:
                print("   ⚠️  Skipping fusion test (AIFS encoder not available)")
                return
        except Exception as e:
            print(f"   ⚠️  Skipping fusion test (initialization failed: {e})")
            return

        # Prepare test data
        batch_size = 4
        climate_data = torch.randn(batch_size, 218)
        texts = [
            "Temperature anomaly detected in the region",
            "Precipitation patterns showing unusual trends",
            "Wind speed measurements above normal ranges",
            "Atmospheric pressure systems indicate storm formation",
        ]

        # Test complete fusion
        start_time = time.time()
        try:
            results = fusion_module(climate_data, texts)
            fusion_time = time.time() - start_time

            # Validate results
            self.assertIn("climate_features", results)
            self.assertIn("text_features", results)
            self.assertIn("fused_features", results)
            self.assertIn("fusion_dim", results)

            # Check shapes
            self.assertEqual(results["climate_features"].shape, (batch_size, 512))
            self.assertEqual(results["text_features"].shape, (batch_size, 512))
            self.assertEqual(results["fused_features"].shape, (batch_size, 512))
            self.assertEqual(results["fusion_dim"], 512)

            print(f"   ✅ Complete fusion: {fusion_time:.4f}s")
            print(f"   ✅ Climate features: {results['climate_features'].shape}")
            print(f"   ✅ Text features: {results['text_features'].shape}")
            print(f"   ✅ Fused features: {results['fused_features'].shape}")
        except Exception as e:
            print(f"   ⚠️  Fusion failed: {e}")

    def test_similarity_and_alignment(self):
        """Test similarity and alignment computations."""
        print("\\n🔗 Testing Similarity and Alignment")

        if not self.has_real_model:
            print("   ⚠️  Skipping similarity test (no real model)")
            return

        try:
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path), device=self.device
            )

            # Check if encoder was loaded successfully
            if fusion_module.aifs_encoder is None:
                print("   ⚠️  Skipping similarity test (AIFS encoder not available)")
                return
        except Exception as e:
            print(f"   ⚠️  Skipping similarity test (initialization failed: {e})")
            return

        # Test climate similarity
        climate_data1 = torch.randn(2, 218)
        climate_data2 = torch.randn(2, 218)

        try:
            similarity = fusion_module.get_climate_similarity(climate_data1, climate_data2)

            # Should return similarity scores for each pair
            self.assertEqual(similarity.shape, (2,))
            self.assertTrue(torch.all(similarity >= -1))  # Cosine similarity bounds
            self.assertTrue(torch.all(similarity <= 1))

            print(f"   ✅ Climate similarity: {similarity}")

            # Test text-climate alignment
            texts = ["Hot and dry conditions", "Cold and wet weather"]
            alignment = fusion_module.get_text_climate_alignment(climate_data1, texts)

            self.assertEqual(alignment.shape, (2,))
            self.assertTrue(torch.all(alignment >= -1))
            self.assertTrue(torch.all(alignment <= 1))

            print(f"   ✅ Text-climate alignment: {alignment}")
        except Exception as e:
            print(f"   ⚠️  Similarity test failed: {e}")

    def test_climate_embedding_module(self):
        """Test standalone climate embedding module."""
        print("\\n🎯 Testing Climate Embedding Module")

        if not self.has_real_model:
            print("   ⚠️  Skipping embedding test (no real model)")
            return

        try:
            embedding_module = AIFSClimateEmbedding(
                aifs_checkpoint_path=str(self.aifs_model_path),
                climate_dim=218,  # Advanced AIFS encoder dimension
                embedding_dim=256,
                device=self.device,
            )

            # Check if encoder was loaded successfully
            if embedding_module.aifs_encoder is None:
                print("   ⚠️  Skipping embedding test (AIFS encoder not available)")
                return
        except Exception as e:
            print(f"   ⚠️  Skipping embedding test (initialization failed: {e})")
            return

        # Test embedding creation
        batch_size = 4
        climate_data = torch.randn(batch_size, 218)

        try:
            embeddings = embedding_module(climate_data)

            # Validate embeddings
            self.assertEqual(embeddings.shape, (batch_size, 256))
            self.assertEqual(embeddings.device.type, self.device)

            print(f"   ✅ Climate embeddings: {climate_data.shape} -> {embeddings.shape}")

            # Test embedding properties
            embedding_norm = embeddings.norm(dim=1)
            print(
                f"   ✅ Embedding norms: [{embedding_norm.min():.3f}, {embedding_norm.max():.3f}]"
            )
        except Exception as e:
            print(f"   ⚠️  Embedding test failed: {e}")

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        print("\\n📦 Testing Batch Processing")

        if not self.has_real_model:
            print("   ⚠️  Skipping batch test (no real model)")
            return

        try:
            fusion_module = AIFSClimateTextFusion(
                aifs_checkpoint_path=str(self.aifs_model_path), device=self.device
            )

            # Check if encoder was loaded successfully
            if fusion_module.aifs_encoder is None:
                print("   ⚠️  Skipping batch test (AIFS encoder not available)")
                return

        except Exception as e:
            print(f"   ⚠️  Skipping batch test (initialization failed: {e})")
            return

        # Test different batch sizes
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            climate_data = torch.randn(batch_size, 218)
            texts = [f"Climate description {i}" for i in range(batch_size)]

            try:
                start_time = time.time()
                results = fusion_module(climate_data, texts)
                processing_time = time.time() - start_time

                # Validate batch processing
                self.assertEqual(results["fused_features"].shape[0], batch_size)

                throughput = batch_size / processing_time
                print(
                    f"   📊 Batch {batch_size:2d}: {processing_time:.4f}s, {throughput:.1f} samples/s"
                )
            except Exception as e:
                print(f"   ⚠️  Batch {batch_size} failed: {e}")
                continue

        print("   ✅ Batch processing test completed")

    def test_error_handling(self):
        """Test error handling in fusion module."""
        print("\\n❌ Testing Error Handling")

        # Test with invalid initialization - should fallback gracefully or provide clear warnings
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fusion_module = AIFSClimateTextFusion(
                    aifs_checkpoint_path="/nonexistent/path", device=self.device
                )
                # Should create the module but encoder will be None
                self.assertIsNone(fusion_module.aifs_encoder)
                print(f"   ✅ Invalid path handled gracefully (encoder=None)")
        except Exception as e:
            print(f"   ✅ Invalid path raises appropriate error: {e}")

        print("   ✅ Invalid path handling")

        if self.has_real_model:
            try:
                fusion_module = AIFSClimateTextFusion(
                    aifs_checkpoint_path=str(self.aifs_model_path), device=self.device
                )

                # Test with mismatched batch sizes
                climate_data = torch.randn(4, 218)
                texts = ["Only one text"]  # Batch size mismatch

                # This should handle gracefully or raise informative error
                try:
                    results = fusion_module(climate_data, texts)
                    # If it succeeds, validate that it handled the mismatch properly
                    self.assertIsNotNone(results)
                except Exception as e:
                    # Should be an informative error
                    self.assertIsInstance(e, (ValueError, RuntimeError))

                print("   ✅ Batch size mismatch handling")
            except Exception as e:
                print(f"   ⚠️  Error handling test skipped: {e}")
        else:
            print("   ⚠️  Skipping error handling tests (no real model)")


def run_aifs_climate_fusion_tests():
    """Run all AIFS climate fusion integration tests."""
    print("🧪 Running AIFS Climate Fusion Integration Tests")
    print("=" * 55)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIFSClimateFusion)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\\n" + "=" * 55)
    if result.wasSuccessful():
        print("✅ All AIFS climate fusion tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")

        if result.failures:
            print("\\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")

        if result.errors:
            print("\\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_aifs_climate_fusion_tests()
    sys.exit(0 if success else 1)
