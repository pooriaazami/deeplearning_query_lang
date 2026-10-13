"""
Unit test for simple_model.dl - validates DL compiler output against reference implementation
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import unittest
from pathlib import Path

# Add project root to path so we can import DL modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dlang.parser.lexer import lex
from src.dlang.parser.parser import parse
from src.dlang.codegen.pytorch_generator import generate_pytorch_code


# Reference implementation (copied from reference_implementations/simple_model.py)
class SimpleModelReference(nn.Module):
    def __init__(self):
        super(SimpleModelReference, self).__init__()
        self.d1 = nn.Linear(784, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        # Input shape: (batch_size, 784)
        x = torch.relu(self.d1(x))
        x = torch.softmax(self.out(x), dim=1)
        return x


class TestSimpleModel(unittest.TestCase):
    """Test class for simple_model.dl"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dl_file_path = "test/basic/simple_model.dl"
        self.reference_model = SimpleModelReference()
        self.reference_model.eval()

        # Parse the DL file and generate PyTorch code
        with open(self.dl_file_path, 'r') as f:
            text = f.read()
        tokens = lex(text)
        ast = parse(tokens)
        pytorch_code = generate_pytorch_code(ast)

        # Execute the generated code to get the model class
        local_namespace = {}
        exec(pytorch_code, {}, local_namespace)

        # Find the model class (should be the only class defined)
        model_class = None
        for name, obj in local_namespace.items():
            if isinstance(obj, type) and hasattr(obj, '__bases__') and len(obj.__bases__) > 0:
                # Check if it inherits from nn.Module
                if nn.Module in obj.__mro__:
                    model_class = obj
                    break

        if model_class is None:
            raise ValueError("Could not find model class in generated code")

        # Instantiate the model
        self.generated_model = model_class()
        self.generated_model.eval()

    def test_models_produce_same_output(self):
        """Test that reference and generated models produce same output"""
        # Create test input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 784)

        # Get outputs from both models
        with torch.no_grad():
            reference_output = self.reference_model(input_tensor)
            generated_output = self.generated_model(input_tensor)

        # Check that outputs are close (within tolerance)
        self.assertTrue(
            torch.allclose(reference_output, generated_output, atol=1e-6),
            f"Outputs differ: max diff = {torch.max(torch.abs(reference_output - generated_output))}"
        )

    def test_models_have_same_structure(self):
        """Test that both models have the same layer structure"""
        # Get state dicts
        ref_state = self.reference_model.state_dict()
        gen_state = self.generated_model.state_dict()

        # Check that both have same keys
        self.assertEqual(
            set(ref_state.keys()),
            set(gen_state.keys()),
            f"State dict keys differ: ref={set(ref_state.keys())}, gen={set(gen_state.keys())}"
        )

        # Check that corresponding layers have same shapes
        for key in ref_state.keys():
            self.assertEqual(
                ref_state[key].shape,
                gen_state[key].shape,
                f"Layer {key} shape mismatch: ref={ref_state[key].shape}, gen={gen_state[key].shape}"
            )


if __name__ == "__main__":
    unittest.main()