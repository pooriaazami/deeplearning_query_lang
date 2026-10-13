# Code generator for the Deep Learning Description Language
# Takes an AST and produces code for the target backend (e.g., PyTorch, TensorFlow, JAX)

from typing import Dict, Any, Optional
from ..parser.ast import *
from .backends.pytorch import PyTorchBackend

class Generator:
    def __init__(self, backend: str = 'pytorch'):
        """
        Initialize the generator with a specific backend.

        Args:
            backend: The target backend ('pytorch', 'tensorflow', 'jax')
        """
        self.backend = backend
        if backend == 'pytorch':
            self.backend_impl = PyTorchBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, ast: Model) -> str:
        """
        Generate code from the AST.

        Args:
            ast: The AST representing the model

        Returns:
            Generated code as a string
        """
        return self.backend_impl.generate(ast)