"""
EMOTIVA / SONORA — Master Test Entry Point
Runs all unit and integration tests across the project.

Usage:
    python test.py
"""
import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_all_tests():
    print("=" * 60)
    print("  SONORA / EMOTIVA — Executing System Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    start_dir = os.path.join(PROJECT_ROOT, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        return 0
    else:
        print(f"  TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
