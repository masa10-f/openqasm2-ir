# openqasm2-ir

Custom IR representation for OpenQASM 2.0

## Overview

**qasm2graphqomb** is a Python library for parsing OpenQASM 2.0 source code and converting it to a custom intermediate representation (IR). It supports quantum circuit manipulation, validation, and integration with GraphQOMB.

## Features

- **OpenQASM 2.0 Parser**: Full OpenQASM 2.0 syntax support
- **Intermediate Representation (IR)**: Efficient representation and manipulation of quantum circuits
- **GraphQOMB Integration**: Seamless conversion to GraphQOMB
- **Type Safety**: Reliable development with static type checking
- **CLI Tools**: Easy command-line usage

## Installation

```bash
pip install qasm2graphqomb
```

## Quick Start

```python
from qasm2.parser import parse_qasm_file
from qasm2.lower import lower_program

# Parse OpenQASM file
ast = parse_qasm_file("circuit.qasm")

# Convert to IR
circuit = lower_program(ast)

# Get circuit information
print(f"Qubits: {circuit.n_qubits}")
print(f"Operations: {len(circuit.ops)}")
```

## Documentation

Full documentation is available in the `docs/` directory. To build the documentation:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

The documentation includes:
- **Tutorial**: Step-by-step guide with examples (Japanese/English)
- **API Reference**: Complete API documentation with Numpy-style docstrings
- **Module Reference**: Detailed module documentation

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy .
pyright

# Linting
ruff check .
```

## License

See LICENSE file for details.
