"""
Test runner for all reference implementation tests using unittest discovery
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run all tests in the reference_implementations directory using unittest discovery"""
    test_dir = Path(__file__).parent

    # Run unittest discovery on the test directory
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", str(test_dir), "-v"],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for all tests
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode

    except subprocess.TimeoutExpired:
        print("Tests timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())