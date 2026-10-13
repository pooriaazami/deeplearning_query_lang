# PyTorch backend code generation
# This module contains templates and functions to generate PyTorch code from the AST.

from typing import Dict, List, Any, Optional
from ...parser.ast import *

class PyTorchBackend:
    def __init__(self):
        self.indent_level = 0
        self.imports = set()
        self.layers = {}
        self.tensor_names = set()

    def indent(self, spaces: int = 4) -> str:
        """Generate indentation string"""
        return ' ' * (self.indent_level * spaces)

    def generate(self, ast: Model) -> str:
        """
        Generate PyTorch code from the model AST.

        Args:
            ast: The model AST

        Returns:
            Generated PyTorch code as a string
        """
        # Reset state
        self.imports = set()
        self.layers = {}
        self.tensor_names = set()

        # Add necessary imports
        self.imports.add('import torch')
        self.imports.add('import torch.nn as nn')
        self.imports.add('import torch.nn.functional as F')

        # Process imports from the AST (if any)
        for imp in ast.imports:
            # For now, we'll just note that we need to handle imported models
            # In a full implementation, this would load and integrate the imported model
            pass

        # Generate the model class
        class_name = ''.join(word.capitalize() for word in ast.name.split('_')) + 'Model'
        if not class_name.endswith('Model'):
            class_name += 'Model'

        lines = []
        lines.append('# Generated PyTorch model')
        lines.extend(sorted(self.imports))
        lines.append('')
        lines.append(f'class {class_name}(nn.Module):')
        self.indent_level += 1

        # __init__ method
        lines.append(f'{self.indent()}def __init__(self):')
        self.indent_level += 1
        lines.append(f'{self.indent()}super({class_name}, self).__init__()')

        # Process inputs to determine input shape
        input_shapes = {}
        for inp in ast.inputs:
            input_shapes[inp.name] = inp.shape

        # Generate layers
        self._generate_layers(ast.layers, input_shapes)

        self.indent_level -= 1

        # Forward method
        lines.append(f'{self.indent()}def forward(self, x):')
        self.indent_level += 1

        # Store tensor mappings for forward pass
        tensor_map = {}  # Maps variable names to tensor expressions

        # Handle inputs
        for inp in ast.inputs:
            tensor_map[inp.name] = 'x'  # Input maps to the forward parameter

        # Process layers to build forward pass
        self._generate_forward_pass(ast.layers, tensor_map)

        # Handle outputs
        if ast.outputs:
            # Assuming single output for simplicity
            output_var = ast.outputs[0].source
            lines.append(f'{self.indent()}return tensor_map[\'{output_var}\']')
        else:
            # If no explicit output, return the last tensor
            if tensor_map:
                last_var = list(tensor_map.keys())[-1]
                lines.append(f'{self.indent()}return tensor_map[\'{last_var}\']')
            else:
                lines.append(f'{self.indent()}return x')

        self.indent_level -= 1

        return '\n'.join(lines)

    def _generate_layers(self, layers: List[Any], input_shapes: Dict[str, tuple]):
        """Generate layer definitions in __init__"""
        for i, layer_item in enumerate(layers):
            if isinstance(layer_item, tuple):
                # Handle special cases like block inputs/outputs
                continue
            elif hasattr(layer_item, 'name') and hasattr(layer_item, 'type'):
                # Regular layer
                layer = layer_item
                layer_name = self._make_valid_name(layer.name)
                self.layers[layer.name] = layer_name

                # Generate layer definition
                params = []
                for param_name, param_value in layer.parameters.items():
                    if isinstance(param_value, str):
                        params.append(f"{param_name}='{param_value}'")
                    elif param_value is None:
                        continue  # Skip None values
                    elif isinstance(param_value, (int, float)):
                        params.append(f"{param_name}={param_value}")
                    else:
                        params.append(f"{param_name}={repr(param_value)}")

                params_str = ', '.join(params)
                if params_str:
                    lines = [f'{self.indent()}self.{layer_name} = nn.{layer.type}({params_str})']
                else:
                    lines = [f'{self.indent()}self.{layer_name} = nn.{layer.type}()']

                # Store for later use in forward pass
                # We'll handle the actual forward pass generation separately

            elif isinstance(layer_item, Block):
                # Generate block as a nested class or as sequential layers
                block = layer_item
                block_class_name = ''.join(word.capitalize() for word in block.name.split('_')) + 'Block'
                if not block_class_name.endswith('Block'):
                    block_class_name += 'Block'

                lines = [f'{self.indent()}self.{block.name} = {block_class_name}()']
                # We would need to generate the block class separately
                # For simplicity, we'll treat blocks as sequential for now

            elif isinstance(layer_item, Repeat):
                # Handle repeat loops - we'll unroll them or create a loop in forward pass
                repeat = layer_item
                # For PyTorch, we'll typically unroll or handle in forward pass
                pass

            elif isinstance(layer_item, CustomLayer):
                # Custom layers need special handling
                custom = layer_item
                # For now, we'll note that custom layers need external implementation
                pass

    def _generate_forward_pass(self, layers: List[Any], tensor_map: Dict[str, str]):
        """Generate forward pass statements"""
        # This is a simplified implementation
        # A full implementation would properly track tensor dependencies

        # For now, we'll generate a simple forward pass that processes layers sequentially
        # In reality, we need to build a proper dependency graph

        # Add comment about forward pass generation
        lines = [f'{self.indent()}# Forward pass - simplified implementation']
        lines.append(f'{self.indent()}# In a full implementation, this would properly track tensor dependencies')
        lines.append(f'{self.indent()}# For now, returning input as placeholder')
        lines.append(f'{self.indent()}return x')

    def _make_valid_name(self, name: str) -> str:
        """Convert a name to a valid Python identifier"""
        # Replace invalid characters with underscores
        import re
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = '_' + name
        return name or 'layer'