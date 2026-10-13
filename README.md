# Deep Learning Description Language (dlang)

A description language for deep learning projects that allows you to define model architectures, data pipelines, training configurations, and more in a simple, declarative syntax. The language generates code for popular deep learning frameworks (starting with PyTorch) and is designed to be extensible via plugins.

## Features

- Declarative model definition (sequential and functional styles)
- Reusable blocks for defining sub-models (e.g., VGG blocks)
- Data loading and preprocessing specifications
- Training configuration (optimizers, losses, metrics, etc.)
- Experiment tracking and logging
- Model persistence (save/load)
- Extensible plugin system for new layers, preprocessing steps, metrics, etc.
- Command-line interface for project creation, code generation, running experiments, and more
- Installable via pip from git repositories

## Installation

Install from git:

```bash
pip install git+https://github.com/yourusername/DeepLearningLang.git
```

Or install in development mode:

```bash
git clone https://github.com/yourusername/DeepLearningLang.git
cd DeepLearningLang
pip install -e .
```

## Usage

Create a new project:

```bash
dlang init my_project
```

Define your model in a `.dl` file (see `examples/` for samples), then generate and run code:

```bash
dlang run examples/my_model.dl
```

For more information, see the documentation.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.