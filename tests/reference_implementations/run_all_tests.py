#!/usr/bin/env python3
"""
Test runner for all reference implementation tests.
Discovers and runs all test files in the reference_implementations directory.
"""
import unittest
import sys
import os

def run_all_tests():
    """Discover and run all tests in the reference_implementations directory."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover all test files (files starting with test_)
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback else 'Unknown'}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1] if traceback else 'Unknown'}")

    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed or had errors.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())