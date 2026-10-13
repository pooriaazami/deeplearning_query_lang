"""
Unit tests for DL (Description Language) models.
Tests that all DL model files can be parsed, validated, and processed correctly.
"""

import os
import unittest
from pathlib import Path

# Import DL processing modules
try:
    from src.dlang.parser.parser import parse_file
    from src.dlang.validation.validator import validate_model
    from src.dlang.codegen.pytorch_generator import generate_pytorch_code
    DL_MODULES_AVAILABLE = True
except ImportError as e:
    DL_MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestDLModels(unittest.TestCase):
    """Test case for DL model files."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - check if DL modules are available."""
        if not DL_MODULES_AVAILABLE:
            raise unittest.SkipTest(f"DL modules not available: {IMPORT_ERROR}")

    def get_all_dl_files(self):
        """Get all .dl files in the test directory and subdirectories."""
        test_dir = Path("D:/programming/projects/DeepLearningLang/test")
        dl_files = []

        # Walk through all subdirectories
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.dl'):
                    full_path = Path(root) / file
                    # Make path relative to test directory for cleaner reporting
                    relative_path = full_path.relative_to(test_dir)
                    dl_files.append((str(full_path), str(relative_path)))

        return dl_files

    def test_dl_files_can_be_parsed(self):
        """Test that all .dl files can be parsed without errors."""
        dl_files = self.get_all_dl_files()

        self.assertGreater(len(dl_files), 0, "No .dl files found in test directory")

        for full_path, relative_path in dl_files:
            with self.subTest(file=relative_path):
                try:
                    # Parse the DL file
                    ast = parse_file(full_path)
                    self.assertIsNotNone(ast, f"AST should not be None for {relative_path}")
                    # Basic check that we got something meaningful
                    self.hasattr(ast, 'model') or self.hasattr(ast, 'blocks') or hasattr(ast, '__dict__')
                except Exception as e:
                    self.fail(f"Failed to parse {relative_path}: {str(e)}")

    def test_dl_files_pass_validation(self):
        """Test that all .dl files pass validation."""
        dl_files = self.get_all_dl_files()

        for full_path, relative_path in dl_files:
            with self.subTest(file=relative_path):
                try:
                    # Parse and validate the DL file
                    ast = parse_file(full_path)
                    is_valid, errors = validate_model(ast)
                    self.assertTrue(is_valid, f"Validation failed for {relative_path}: {errors}")
                except Exception as e:
                    self.fail(f"Failed to validate {relative_path}: {str(e)}")

    @unittest.skipIf(not hasattr(unittest.TestCase, 'skipIf'), "Code generation test skipped if not implemented")
    def test_dl_files_can_generate_code(self):
        """Test that all .dl files can generate PyTorch code (if codegen is available)."""
        # Check if code generation is available
        try:
            from src.dlang.codegen.pytorch_generator import generate_pytorch_code
            codegen_available = True
        except ImportError:
            codegen_available = False

        if not codegen_available:
            self.skipTest("Code generation not available")

        dl_files = self.get_all_dl_files()

        for full_path, relative_path in dl_files:
            with self.subTest(file=relative_path):
                try:
                    # Parse the DL file
                    ast = parse_file(full_path)
                    # Generate PyTorch code
                    code = generate_pytorch_code(ast)
                    self.assertIsInstance(code, str, f"Generated code should be a string for {relative_path}")
                    self.assertGreater(len(code.strip()), 0, f"Generated code should not be empty for {relative_path}")
                    # Basic sanity check - should contain class definition or function
                    self.assertIn('class', code) or self.assertIn('def', code) or self.assertIn('import', code)
                except Exception as e:
                    self.fail(f"Failed to generate code for {relative_path}: {str(e)}")


if __name__ == '__main__':
    unittest.main()