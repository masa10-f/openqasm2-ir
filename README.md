# openqasm2-ir

Custom IR representation for OpenQASM 2.0

## Overview

**qasm2graphqomb** is a Python library for parsing OpenQASM 2.0 source code and converting it to a custom intermediate representation (IR). It supports quantum circuit manipulation, validation, and integration with GraphQOMB.

## Features

- **OpenQASM 2.0 Parser**: Full OpenQASM 2.0 syntax support
- **Intermediate Representation (IR)**: Efficient representation and manipulation of quantum circuits
- **GraphQOMB Integration**: Seamless conversion to GraphQOMB
- **QASM Optimization**: Circuit optimization using pytket
- **Type Safety**: Reliable development with static type checking
- **CLI Tools**: Easy command-line usage

## Installation

```bash
pip install qasm2graphqomb
```

## Quick Start

### Parsing and IR Conversion

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

### QASM Optimization with pytket

```bash
# Install optimizer dependencies
pip install -e ".[optimizer]"

# Optimize a QASM file
tket-optimize-qasm input.qasm output_optimized.qasm

# Or use as a Python module
python tket_opt_qasm.py input.qasm output_optimized.qasm
```

See [optimizer/README.md](optimizer/README.md) for detailed usage and examples.

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

# Install optimizer dependencies
pip install -e ".[optimizer]"

# Run tests
pytest

# Type checking
mypy .
pyright

# Linting
ruff check .
```

## Optional Dependencies

- **optimizer**: Install pytket for QASM circuit optimization
  ```bash
  pip install -e ".[optimizer]"
  ```
- **dev**: Install development tools (pytest, mypy, pyright, ruff)
- **docs**: Install documentation dependencies (Sphinx)

## License

See LICENSE file for details.
