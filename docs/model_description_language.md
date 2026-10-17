# Deep Learning Description Language (DL) - Developer Documentation

## Overview

The Deep Learning Description Language (DL) is a domain-specific language designed for defining neural network architectures in a clear, concise, and reusable manner. It allows developers to describe model architectures using high-level constructs while generating executable PyTorch code underneath.

## Keywords and Syntax

### Core Language Keywords

| Keyword | Description | Example |
|---------|-------------|---------|
| `MODEL` | Defines a model | `MODEL my_model` |
| `END` | Marks the end of a model or block | `END` |
| `INPUT` | Defines model input | `INPUT(shape=(784,), name='input')` |
| `OUTPUT` | Defines model output | `OUTPUT = tensor_name` |
| `BLOCK` | Defines a reusable block | `BLOCK my_block(params)...END` |
| `REPEAT` | Creates a repetition loop | `REPEAT 3 times: ... END` |
| `times` | Part of REPEAT syntax | `REPEAT 5 times:` |
| `IMPORT` | Imports another DL file | `IMPORT "components.dl"` |

### Layer Types and Operations

The following layer types are supported in the language (case-sensitive as shown):

- **Convolutional Layers**: `Conv2D`, `Conv1D`, `Conv3D`
- **Pooling Layers**: `MaxPool2D`, `MaxPool1D`, `MaxPool3D`, `AveragePool2D`, `GlobalAveragePooling2D`
- **Recurrent Layers**: `LSTM`, `GRU`, `RNN`
- **Dense Layers**: `Dense` (equivalent to Linear)
- **Normalization**: `BatchNormalization`, `LayerNormalization`
- **Dropout**: `Dropout`
- **Activation**: `Activation`
- **Merge Operations**: `Add`, `Concatenate`, `ElementwiseMultiply`, `Multiply`, `Subtract`
- **Reshape Operations**: `Reshape`, `Flatten`
- **Custom Layers**: `CUSTOM`

### Literals and Identifiers

| Type | Syntax | Examples |
|------|--------|----------|
| **Strings** | `"text"` or `'text'` | `"relu"`, `'my_layer'` |
| **Numbers** | Digits with optional decimal | `128`, `0.1`, `3.14` |
| **Booleans** | `True` or `False` | `True`, `False` |
| **Identifiers** | Letters, digits, underscore | `input`, `dense1`, `batch_norm_2` |
| **None** | `None` | Used in shapes for dynamic dimensions |

### Operators

| Operator | Description |
|----------|-------------|
| `=` | Assignment |
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `()` | Function calls and grouping |
| `[]` | List indexing |
| `,` | Parameter separation |
| `:` | Used in REPEAT statements |
| `.` | Attribute access |

## Language Features

### 1. Model Definition

```dl
MODEL model_name
    INPUT(shape=(input_dim,), name='input')
    # Layers go here
    OUTPUT = output_tensor
END
```

### 2. Functional API Style

The language supports defining models with multiple inputs and outputs:

```dl
MODEL functional_model
    INPUT(shape=(32,), name='input_a')
    INPUT(shape=(32,), name='input_b')
    # Processing layers
    combined = Concatenate([proc_A, proc_B])
    OUTPUT = Dense([combined], units=10, activation='softmax')
END
```

### 3. Reusable Blocks (Sub-models)

Blocks allow defining reusable components that can be parameterized and instantiated multiple times:

```dl
# Define a reusable block
BLOCK conv_block(filters, kernel_size, input_name, prefix='')
    INPUT(name=input_name)
    conv1 = Conv2D([INPUT], filters=filters, kernel_size=kernel_size, 
                   activation='relu', padding='same', name=prefix+'conv1')
    conv2 = Conv2D([conv1], filters=filters, kernel_size=kernel_size, 
                   activation='relu', padding='same', name=prefix+'conv2')
    OUTPUT = conv2
END

# Use the block in a model
MODEL vgg_like
    INPUT(shape=(224,224,3), name='input')
    block1_out = conv_block(filters=64, kernel_size=3, input_name='input', prefix='block1_')
    pool1 = MaxPool2D([block1_out], pool_size=2, strides=2, name='pool1')
    # ... more layers
END
```

### 4. Repetition Constructs

```dl
MODEL repeated_model
    INPUT(shape=(784,), name='input')
    x = INPUT
    REPEAT 3 times:
        x = Dense([x], units=128, activation='relu')
    OUTPUT = Dense([x], units=10, activation='softmax')
END
```

### 5. Layer Connections

Layers connect by referencing previous layer outputs:

```dl
MODEL connection_example
    INPUT(shape=(784,), name='input')
    # Reference previous layers by their variable names
    dense1 = Dense([INPUT], units=128, activation='relu', name='fc1')
    dense2 = Dense([dense1], units=64, activation='relu', name='fc2')
    OUTPUT = Dense([dense2], units=10, activation='softmax', name='output')
END
```

### 6. Activation Functions

Activation functions can be specified as strings:

```dl
dense = Dense([input], units=64, activation='relu')
# Or as separate activation layers
activation = Activation([dense], activation='relu')
```

Supported activations include: `'relu'`, `'sigmoid'`, `'tanh'`, `'softmax'`, `'linear'`, `'elu'`, `'selu'`, etc.

## Implemented Features

Based on the code analysis, the following features have been implemented:

### ✅ Core Language Features

- **Model Architecture Definition**
  - Sequential model definition
  - Functional API-like definition (multiple inputs/outputs)
  - Reusable blocks with parameterization
  - Block composition (blocks containing other blocks)
  - Input/output shape specification
  
- **Layer Types Supported**
  - Dense (fully connected) layers
  - Conv1D, Conv2D, Conv3D layers
  - RNN, LSTM, GRU recurrent layers
  - Dropout layers
  - BatchNormalization layers
  - Activation layers
  - Pooling layers (MaxPool, AveragePool, GlobalPool)
  - Merge layers (Concatenate, Add, Subtract, Multiply)
  - Reshape and Flatten layers

- **Control Flow**
  - REPEAT loops for iterative layer application

- **Abstraction Mechanisms**
  - BLOCK definitions for reusable components
  - Parameterized blocks
  - Block instantiation with parameter passing

### 🔧 Code Generation Capabilities

- **PyTorch Backend**
  - Generates idiomatic PyTorch nn.Module code
  - Proper layer mapping (Dense → nn.Linear, Conv2D → nn.Conv2d, etc.)
  - Automatic parameter extraction and passing
  - Forward pass generation from layer connections
  - Import handling for torch and torch.nn

### 📝 Sample Models

#### 1. Simple Feedforward Network
```dl
MODEL simple_model
    INPUT(shape=(784,), name='input')
    d1 = Dense([input], units=128, activation='relu', name='dense1')
    out = Dense([d1], units=10, activation='softmax', name='output')
    OUTPUT = out
END
```

#### 2. Functional Model with Multiple Inputs
```dl
MODEL functional_model
    INPUT(shape=(32,), name='input_a')
    INPUT(shape=(32,), name='input_b')
    proc_A = Dense([INPUT], units=64, activation='relu', name='proc_A')
    proc_B = Dense([INPUT], units=64, activation='relu', name='proc_B')
    combined = Concatenate([proc_A, proc_B], name='concat')
    OUTPUT = Dense([combined], units=10, activation='softmax', name='output')
END
```

#### 3. CNN with Reusable Blocks
```dl
# Define a reusable conv block
BLOCK conv_block(filters, input_name, prefix='')
    INPUT(name=input_name)
    conv1 = Conv2D([INPUT], filters=filters, kernel_size=3, 
                   activation='relu', padding='same', name=prefix+'conv1')
    conv2 = Conv2D([conv1], filters=filters, kernel_size=3, 
                   activation='relu', padding='same', name=prefix+'conv2')
    OUTPUT = conv2
END

MODEL cnn_model
    INPUT(shape=(32,32,3), name='input')
    
    # First block
    block1_out = conv_block(filters=32, input_name='input', prefix='block1_')
    pool1 = MaxPool2D([block1_out], pool_size=2, name='pool1')
    
    # Second block
    block2_out = conv_block(filters=64, input_name=pool1, prefix='block2_')
    pool2 = MaxPool2D([block2_out], pool_size=2, name='pool2')
    
    # Classifier
    flat = Flatten([pool2], name='flatten')
    fc1 = Dense([flat], units=128, activation='relu', name='fc1')
    OUTPUT = Dense([fc1], units=10, activation='softmax', name='output')
END
```

#### 4. RNN Sequence Model
```dl
MODEL sequence_model
    INPUT(shape=(None, 100), name='sequences')  # Variable length sequences
    # Process sequence with LSTM
    lstm_out = LSTM([INPUT], units=128, return_sequences=False)
    # Apply dropout for regularization
    dropout = Dropout([lstm_out], rate=0.5)
    # Final classification
    OUTPUT = Dense([dropout], units=10, activation='softmax', name='predictions')
END
```

## Usage Guidelines

### Naming Conventions

- Layer types use PascalCase (e.g., `Conv2D`, `LSTM`, `BatchNormalization`)
- Variable names use snake_case (e.g., `input`, `dense1`, `conv_block`)
- String literals can use either single or double quotes
- Layer names in assignments become variable references for connecting layers

### Shape Specification

- Input shapes use tuple notation: `(dim1, dim2, ...)`
- Use `None` for variable dimensions (e.g., batch size or sequence length)
- Shape elements are integers or `None`

### Parameter Passing

- Parameters are passed as keyword arguments: `param_name=value`
- Common parameters include:
  - `units`: Number of neurons in dense layers
  - `filters`: Number of filters in convolutional layers
  - `kernel_size`: Size of convolutional kernel
  - `activation`: Activation function name
  - `padding`: `'valid'` or `'same'`
  - `strides`: Stride size for convolution/pooling
  - `pool_size`: Size of pooling window
  - `rate`: Dropout rate (0-1)
  - `return_sequences`: Boolean for RNN layers

## Extensibility

The language is designed to be extensible through:

1. **Custom Layers**: Using the `CUSTOM` keyword to define bespoke operations
2. **Plugin System**: Additional layer types can be added via plugins
3. **Backend Support**: While PyTorch is the initial backend, the architecture supports others

## Generated Code Structure

When processed by the DL compiler, models generate PyTorch code with this structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelName(nn.Module):
    def __init__(self):
        super(ModelName, self).__init__()
        # Layer definitions generated here
        self.layer1 = nn.Linear(...)
        self.layer2 = nn.Conv2d(...)
        # ... more layers
    
    def forward(self, x):
        # Forward pass generated from layer connections
        tensor_map = {}
        tensor_map['input'] = x
        tensor_map['layer1_out'] = self.layer1(tensor_map['input'])
        tensor_map['layer2_out'] = self.layer2(tensor_map['layer1_out'])
        # ... more computations
        return tensor_map['output_name']
```

## Limitations and Notes

1. **Case Sensitivity**: Layer type names are case-sensitive (use exact casing as shown)
2. **Shape Inference**: Some shape dimensions may need to be specified explicitly
3. **Block Handling**: Blocks are currently flattened in the generated code
4. **Repeat Loops**: REPEAT constructs are unrolled in the current implementation
5. **Single Output**: The current implementation assumes a single model output

## Getting Started

To use the DL language:

1. Write your model architecture in a `.dl` file using the syntax described above
2. Process the file with the DL compiler to generate PyTorch code
3. Use the generated code in your training/inference pipeline

Example workflow:
```bash
# Parse and generate code
dlang parse model.dl --output generated_model.py

# Or compile and run directly
dlang run model.dl --backend pytorch --device cuda
```

This documentation covers the currently implemented features of the Deep Learning Description Language. As the language evolves, additional features and layer types will be added based on community needs and technological advances.