# Contributing to Deep Learning Description Language

Thank you for your interest in contributing to DL! We welcome contributions from the community.

## How to Contribute

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Ensure your code follows the project's coding standards
6. Add tests for any new functionality
7. Run the test suite to ensure everything passes
8. Commit and push your changes
9. Submit a pull request

## Development Setup

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepLearningLang.git
cd DeepLearningLang

# Install in development mode with all extras
pip install -e .[pytorch,tensorflow,docs]
```

## Running Tests

To run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=src/dlang
```

## Documentation

To build the documentation locally:

```bash
# Install docs dependencies if not already installed
pip install .[docs]

# Build the documentation
mkdocs build

# Serve the documentation locally for preview
mkdocs serve
```

## Code Style

Please follow the existing code style in the project. We use:
- PEP 8 for Python code formatting
- Clear, descriptive variable and function names
- Comprehensive docstrings for public APIs
- Type hints where appropriate

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Any relevant code snippets or error messages

Thank you for contributing to DL!