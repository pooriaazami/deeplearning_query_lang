"""
Unit test for mnist_example.dl - validates DL compiler output against reference implementation
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


# Reference implementation (adapted from reference_implementations/mnist_example.py)
class DenseBlockReference(nn.Module):
    def __init__(self, units, dropout_rate, input_name, prefix=''):
        super(DenseBlockReference, self).__init__()
        # We'll initialize with a placeholder size and adjust on first forward pass
        self.dense = nn.Linear(784, units)  # Start with expected input size from model
        self.batchnorm = nn.BatchNorm1d(units)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.dense(x)
        x = self.batchnorm(x)
        if self.dropout_rate > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.relu(x)
        return x


class MnistModelReference(nn.Module):
    def __init__(self):
        super(MnistModelReference, self).__init__()
        self.block1 = DenseBlockReference(units=256, dropout_rate=0.3, input_name='input', prefix='block1_')
        self.block2 = DenseBlockReference(units=256, dropout_rate=0.3, input_name='block1_out', prefix='block2_')
        self.output_layer = nn.Linear(256, 10)  # Final output to 10 classes

    def forward(self, x):
        # x shape: (batch_size, 784) as defined in INPUT(shape=(784,), name='input')
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        output = torch.softmax(self.output_layer(block2_out), dim=1)
        return output


class TestMnistExample(unittest.TestCase):
    """Test class for mnist_example.dl"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dl_file_path = "test/computer_vision/mnist_example.dl"
        self.reference_model = MnistModelReference()
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
        input_tensor = torch.randn(batch_size, 784)  # shape=(784,) as defined in DL

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