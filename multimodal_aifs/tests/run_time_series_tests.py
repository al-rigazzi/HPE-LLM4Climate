#!/usr/bin/env python3
"""
AIFS Time Series Tokenizer Test Runner

This script runs comprehensive tests for the AIFSTimeSeriesTokenizer,
including unit tests, integration tests, and performance benchmarks.

Usage:
    python multimodal_aifs/tests/run_time_series_tests.py
    python multimodal_aifs/tests/run_time_series_tests.py --quick
    python multimodal_aifs/tests/run_time_series_tests.py --benchmarks-only
"""

import argparse
import sys
import time
import unittest
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_unit_tests(verbosity=2):
    """Run unit tests for time series tokenizer."""
    print("Running Unit Tests")
    print("=" * 50)

    try:
        from multimodal_aifs.tests.unit.test_aifs_time_series_tokenizer import (
            TestAIFSTimeSeriesTokenizer,
        )

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestAIFSTimeSeriesTokenizer)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError as e:
        print(f"Could not import unit tests: {e}")
        return False


def run_integration_tests(verbosity=2):
    """Run integration tests for time series tokenizer."""
    print("\\n🔗 Running Integration Tests")
    print("=" * 50)

    try:
        from multimodal_aifs.tests.integration.test_time_series_integration import (
            TestTimeSeriesIntegration,
        )

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestTimeSeriesIntegration)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError as e:
        print(f"Could not import integration tests: {e}")
        return False


def run_performance_benchmarks(verbosity=2):
    """Run performance benchmarks for time series tokenizer."""
    print("\\nRunning Performance Benchmarks")
    print("=" * 50)

    try:
        from multimodal_aifs.tests.benchmarks.test_time_series_performance import (
            TimeSeriesPerformanceBenchmark,
        )

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TimeSeriesPerformanceBenchmark)

        # Run benchmarks
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError as e:
        print(f"Could not import performance benchmarks: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking Dependencies")
    print("=" * 50)

    required_packages = [
        "torch",
        "numpy",
    ]

    optional_packages = [
        "matplotlib",
    ]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   {package}")
        except ImportError:
            missing_required.append(package)
            print(f"   {package} (required)")

    for package in optional_packages:
        try:
            __import__(package)
            print(f"   {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"   {package} (optional)")

    if missing_required:
        print(f"\\nMissing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\\nMissing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))

    return True


def check_tokenizer_availability():
    """Check if the time series tokenizer is available."""
    print("\\nChecking Tokenizer Availability")
    print("=" * 50)

    try:
        from multimodal_aifs.utils.aifs_time_series_tokenizer import AIFSTimeSeriesTokenizer

        # Try to create a tokenizer
        tokenizer = AIFSTimeSeriesTokenizer(device="cpu")
        print("   AIFSTimeSeriesTokenizer imported successfully")

        # Try to get info
        info = tokenizer.get_tokenizer_info()
        print(f"   Tokenizer info: {info['temporal_modeling']} mode")

        return True

    except Exception as e:
        print(f"   Tokenizer not available: {e}")
        return False


def generate_test_report(unit_success, integration_success, benchmark_success, total_time):
    """Generate a comprehensive test report."""
    print("\\nTest Report")
    print("=" * 60)

    # Summary
    total_tests = 3
    passed_tests = sum([unit_success, integration_success, benchmark_success])

    print(f"Overall Results: {passed_tests}/{total_tests} test suites passed")
    print(f"Total execution time: {total_time:.2f} seconds")
    print()

    # Detailed results
    test_results = [
        ("Unit Tests", unit_success),
        ("Integration Tests", integration_success),
        ("Performance Benchmarks", benchmark_success),
    ]

    for test_name, success in test_results:
        status = "PASSED" if success else "FAILED"
        print(f"  {test_name}: {status}")

    print()

    # Recommendations
    if not unit_success:
        print("Unit test failures indicate issues with core functionality.")
        print("   Check tokenizer implementation and basic operations.")

    if not integration_success:
        print("Integration test failures indicate issues with multimodal workflows.")
        print("   Check fusion patterns and end-to-end pipelines.")

    if not benchmark_success:
        print("Benchmark failures indicate performance issues.")
        print("   Check system resources and optimization settings.")

    if passed_tests == total_tests:
        print("All tests passed! Time series tokenizer is ready for production.")
    else:
        print("Some tests failed. Review the output above for details.")

    print("=" * 60)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run AIFS Time Series Tokenizer tests")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--benchmarks-only", action="store_true", help="Run only performance benchmarks"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Suppress warnings unless verbose
    if not args.verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    print("AIFS Time Series Tokenizer Test Suite")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print()

    start_time = time.time()

    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Exiting.")
        return 1

    # Check tokenizer availability
    if not check_tokenizer_availability():
        print("Tokenizer availability check failed. Exiting.")
        return 1

    # Determine which tests to run
    run_unit = not (args.integration_only or args.benchmarks_only)
    run_integration = not (args.unit_only or args.benchmarks_only)
    run_benchmarks = not (args.unit_only or args.integration_only or args.quick)

    # Run tests
    unit_success = True
    integration_success = True
    benchmark_success = True

    if run_unit:
        unit_success = run_unit_tests(verbosity)

    if run_integration:
        integration_success = run_integration_tests(verbosity)

    if run_benchmarks:
        benchmark_success = run_performance_benchmarks(verbosity)

    # Generate report
    total_time = time.time() - start_time
    generate_test_report(unit_success, integration_success, benchmark_success, total_time)

    # Return appropriate exit code
    if unit_success and integration_success and benchmark_success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
