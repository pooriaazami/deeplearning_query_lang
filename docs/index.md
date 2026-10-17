# Deep Learning Description Language (DL)

Welcome to the documentation for the Deep Learning Description Language (DL), a domain-specific language for defining neural network architectures.

## Overview

DL allows you to describe deep learning models in a clear, concise syntax that gets compiled to executable PyTorch (or other backend) code. With DL, you can:

- Define model architectures using intuitive, high-level constructs
- Create reusable building blocks with the `BLOCK` mechanism
- Specify complex models with multiple inputs and outputs
- Leverage powerful abstraction capabilities to reduce boilerplate
- Generate production-ready code with a single command

## Documentation

- [Model Description Language Reference](model_description_language.md) - Complete guide to DL syntax, keywords, and features
- [Features Overview](FEATURES.md) - Planned and implemented features roadmap
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Details on validated language constructs
- [Getting Started](README.md) - Introduction and basic usage

## Getting Started

To get started with DL:

1. Install the package: `pip install -e .`
2. Install backend dependencies: `pip install .[pytorch]` or `pip install .[tensorflow]`
3. For documentation generation: `pip install .[docs]`
4. Write your model in a `.dl` file
5. Generate code: `dlang parse model.dl --output model.py`

## Examples

See the [test_cases](./test_cases/) directory for working examples of:
- Simple feedforward networks
- Convolutional neural networks
- Recurrent neural networks
- Models with reusable blocks
- Functional API-style models

## Community

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on how to contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.