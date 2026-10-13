"""
Unit test for functional_model.dl - validates DL compiler output against reference implementation
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


# Reference implementation (adapted from reference_implementations/functional_model.py)
class InputStreamReference(nn.Module):
    def __init__(self, units, input_name, prefix=''):
        super(InputStreamReference, self).__init__()
        # Hardcoded input size 32 based on the model definition
        self.dense = nn.Linear(32, units)
        self.units = units

    def forward(self, x):
        return torch.relu(self.dense(x))


class FunctionalModelReference(nn.Module):
    def __init__(self):
        super(FunctionalModelReference, self).__init__()
        self.proc_A = InputStreamReference(units=64, input_name='input_a', prefix='stream_a_')
        self.proc_B = InputStreamReference(units=64, input_name='input_b', prefix='stream_b_')
        self.output_layer = nn.Linear(128, 10)  # 64 + 64 = 128 input features

    def forward(self, input_a, input_b):
        out_A = self.proc_A(input_a)
        out_B = self.proc_B(input_b)
        combined = torch.cat([out_A, out_B], dim=1)
        output = torch.softmax(self.output_layer(combined), dim=1)
        return output


class TestFunctionalModel(unittest.TestCase):
    """Test class for functional_model.dl"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dl_file_path = "test/basic/functional_model.dl"
        self.reference_model = FunctionalModelReference()
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
        # Create test inputs
        batch_size = 4
        input_a = torch.randn(batch_size, 32)  # shape=(32,) as defined in DL
        input_b = torch.randn(batch_size, 32)  # shape=(32,) as defined in DL

        # Get outputs from both models
        with torch.no_grad():
            reference_output = self.reference_model(input_a, input_b)
            generated_output = self.generated_model(input_a, input_b)

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