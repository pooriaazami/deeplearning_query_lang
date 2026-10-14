# PyTorch backend code generation
# This module contains templates and functions to generate PyTorch code from the AST.

from typing import Dict, List, Any, Optional
from ...parser.ast import (
    Model, Input, Output, Assignment, FunctionCall,
    Number, String, Variable, Boolean,
    Block, Repeat, CustomLayer
)

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
        self._generate_layers(ast.layers, input_shapes, lines)

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
        self._generate_forward_pass(ast.layers, tensor_map, lines)

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

    def _generate_layers(self, layers: List[Any], input_shapes: Dict[str, tuple], lines: List[str]):
        """Generate layer definitions in __init__"""
        for i, layer_item in enumerate(layers):
            if isinstance(layer_item, tuple):
                # Handle special cases like block inputs/outputs
                continue
            elif hasattr(layer_item, 'target') and hasattr(layer_item, 'expression'):
                # Assignment object - this represents a layer definition or variable assignment
                assignment = layer_item
                target_name = assignment.target
                expression = assignment.expression

                # Only process FunctionCall expressions as layers (DENSE, Conv2D, etc.)
                # Variable expressions (like INPUT) are handled in forward pass
                if isinstance(expression, FunctionCall):
                    func_call = expression
                    # Extract function details
                    func_name = func_call.name  # e.g., 'DENSE'

                    # Convert to valid Python layer name
                    layer_name = self._make_valid_name(target_name)
                    self.layers[target_name] = layer_name

                    # Generate layer definition from function call
                    # Start with the base class from func_name
                    pytorch_class = self._map_to_pytorch_class(func_name)

                    # Collect parameters from args and kwargs
                    params = []

                    # Process positional arguments (args)
                    for arg in func_call.args:
                        param_str = self._extract_value(arg)
                        if param_str is not None:
                            params.append(param_str)

                    # Process keyword arguments (kwargs)
                    for kwarg_name, kwarg_value in func_call.kwargs.items():
                        param_str = self._extract_value(kwarg_value)
                        if param_str is not None:
                            params.append(f"{kwarg_name}={param_str}")

                    params_str = ', '.join(params)
                    if params_str:
                        lines.append(f'{self.indent()}self.{layer_name} = nn.{pytorch_class}({params_str})')
                    else:
                        lines.append(f'{self.indent()}self.{layer_name} = nn.{pytorch_class}()')

            elif isinstance(layer_item, Block):
                # Generate block as a nested class or as sequential layers
                block = layer_item
                block_class_name = ''.join(word.capitalize() for word in block.name.split('_')) + 'Block'
                if not block_class_name.endswith('Block'):
                    block_class_name += 'Block'

                lines.append(f'{self.indent()}self.{block.name} = {block_class_name}()')
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

    def _extract_value(self, ast_node) -> Optional[str]:
        """Extract a string representation of a value from an AST node"""
        if isinstance(ast_node, Number):
            return str(ast_node.value)
        elif isinstance(ast_node, String):
            # Return the string value WITH quotes for code generation
            return repr(ast_node.value)
        elif isinstance(ast_node, Boolean):
            # Return the boolean value as a Python literal
            return str(ast_node.value)
        elif isinstance(ast_node, Variable):
            # This refers to a variable/tensor - handled in forward pass, not layer definition
            return None
        else:
            # For other types, try to get a string representation
            try:
                return str(ast_node.value) if hasattr(ast_node, 'value') else repr(ast_node)
            except:
                return repr(ast_node)

    def _generate_forward_pass(self, layers: List[Any], tensor_map: Dict[str, str], lines: List[str]):
        """Generate forward pass statements"""
        # Process each layer to build the forward pass
        for layer_item in layers:
            if isinstance(layer_item, tuple):
                # Handle special cases like block inputs/outputs
                continue
            elif hasattr(layer_item, 'target') and hasattr(layer_item, 'expression'):
                # Assignment object - this represents a layer application or variable assignment
                assignment = layer_item
                target_name = assignment.target
                expression = assignment.expression

                if isinstance(expression, FunctionCall):
                    # This is a layer/function call (DENSE, Conv2D, etc.)
                    func_call = expression

                    # Get input tensor names from args
                    input_tensors = []
                    for arg in func_call.args:
                        tensor_str = self._extract_tensor_reference(arg, tensor_map)
                        if tensor_str is not None:
                            input_tensors.append(tensor_str)

                    # Generate the layer call
                    layer_name = self._make_valid_name(target_name)
                    if len(input_tensors) == 1:
                        tensor_map[target_name] = f'self.{layer_name}({input_tensors[0]})'
                    elif len(input_tensors) > 1:
                        tensor_map[target_name] = f'self.{layer_name}({", ".join(input_tensors)})'
                    else:
                        # No tensor inputs - this shouldn't happen for layers, but handle gracefully
                        tensor_map[target_name] = f'self.{layer_name}()'

                elif isinstance(expression, Variable):
                    # This is a variable reference (like INPUT, or referencing a previous layer)
                    var_name = expression.name
                    if var_name in tensor_map:
                        tensor_map[target_name] = tensor_map[var_name]
                    elif var_name == 'INPUT':
                        # Special case: INPUT token refers to the model's input
                        # We assume there is only one input and use the forward parameter 'x'
                        tensor_map[target_name] = 'x'
                    else:
                        # Unknown variable - for INPUT, this should be handled by the input mapping
                        # For other variables, it might be an error
                        tensor_map[target_name] = var_name  # Fallback

            elif isinstance(layer_item, Repeat):
                # Handle repeat loops - unroll for now
                repeat = layer_item
                # For simplicity, we'll skip complex loop handling in this implementation
                # A full implementation would need to unroll or create actual loops
                pass

            # Handle activation functions and other operations that might be in layers
            # For now, we assume they're handled as part of layer definitions

    def _extract_tensor_reference(self, ast_node, tensor_map: Dict[str, str]) -> Optional[str]:
        """Extract a tensor reference from an AST node, returning the mapped tensor name"""
        if isinstance(ast_node, Variable):
            if ast_node.name in tensor_map:
                return tensor_map[ast_node.name]
            else:
                # Unknown variable - return as-is for now
                return ast_node.name
        else:
            # For literals and other values, extract their string representation
            value_str = self._extract_value(ast_node)
            return value_str

    def _map_to_pytorch_class(self, func_name: str) -> str:
        """Map DL function names to PyTorch class names"""
        func_name_upper = func_name.upper()
        mapping = {
            'DENSE': 'Linear',
            'CONV1D': 'Conv1d',
            'CONV2D': 'Conv2d',
            'CONV3D': 'Conv3d',
            'GRU': 'GRU',
            'LSTM': 'LSTM',
            'DROPOUT': 'Dropout',
            'BATCHNORM': 'BatchNorm1d',
            'MAXPOOL1D': 'MaxPool1d',
            'MAXPOOL2D': 'MaxPool2d',
            'MAXPOOL3D': 'MaxPool3d',
            'AVGPOOL1D': 'AvgPool1d',
            'AVGPOOL2D': 'AvgPool2d',
            'AVGPOOL3D': 'AvgPool3d',
            'GLOBALAVERAGEPOOLING1D': 'GlobalAvgPool1d',
            'GLOBALAVERAGEPOOLING2D': 'GlobalAvgPool2d',
            'GLOBALAVERAGEPOOLING3D': 'GlobalAvgPool3d',
            'GLOBALMAXPOOLING1D': 'GlobalMaxPool1d',
            'GLOBALMAXPOOLING2D': 'GlobalMaxPool2d',
            'GLOBALMAXPOOLING3D': 'GlobalMaxPool3d',
        }
        return mapping.get(func_name_upper, func_name)  # Default to original name if not found

    def _make_valid_name(self, name: str) -> str:
        """Convert a name to a valid Python identifier"""
        # Replace invalid characters with underscores
        import re
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = '_' + name
        return name or 'layer'