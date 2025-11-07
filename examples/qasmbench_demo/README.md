# QASMBench to GraphQOMB Conversion Demo

## Overview
This demo showcases how a selection of QASMBench OpenQASM 2 circuits are progressively lowered into the GraphQOMB representation using the `openqasm2-ir` tooling plus the GraphQOMB adapter utilities.

The demo reads QASM files directly from the QASMBench repository to avoid license issues with copying circuit files.

## Requirements
- Python 3.10 or newer
- `openqasm2-ir` (provides the conversion pipeline)
- `graphqomb` (supplies the target circuit data structures)
- QASMBench repository (cloned separately)

Install the dependencies:

```bash
# Install graphqomb from PyPI
pip install graphqomb

# Install openqasm2-ir in development mode
cd path/to/openqasm2-ir
pip install -e .

# Clone QASMBench repository if not already available
git clone https://github.com/pnnl/QASMBench.git
```

## Demo Circuits
| Circuit        | Description                        |
|----------------|-------------------------------------|
| `deutsch_n2`   | Deutsch algorithm (2 qubits)        |
| `fredkin_n3`   | Fredkin gate (3 qubits)             |
| `bell_n4`      | Bell state preparation (4 qubits)   |
| `grover_n2`    | Grover's algorithm (2 qubits)       |
| `cat_state_n4` | Cat state generation (4 qubits)     |
| `iswap_n2`     | iSWAP gate (2 qubits)               |

These circuits are read directly from the QASMBench repository, not copied locally.

## Usage
Navigate to the demo directory and run the script:

```bash
cd openqasm2-ir/examples/qasmbench_demo

# Set QASMBENCH_PATH environment variable (required)
export QASMBENCH_PATH=/path/to/QASMBench

# If openqasm2-ir is installed (pip install -e .), simply run:
python demo_converter.py

# Or if running from the repository without installation:
PYTHONPATH=../../:$PYTHONPATH python demo_converter.py
```

**Important**: The `QASMBENCH_PATH` environment variable is required and must point to your local QASMBench repository clone.

The script automatically:
1. Locates the six demo circuits in QASMBench/small/
2. Converts each QASM file to a GraphQOMB circuit using `qasm_file_to_graphqomb()`
3. Displays circuit statistics (qubit count, gate count, gate types)

## Expected Output
A successful conversion displays statistics for each circuit:

```
QASMBench → GraphQOMB Conversion Demo
====================================
Circuits directory: <path-to-QASMBench>/small
Gate mappings file: <openqasm2-ir-root>/gates/gates.yaml

=== deutsch_n2 ===
Circuit Statistics — deutsch_n2
-------------------------------
Qubits      : 2
Total gates : 11
Gate breakdown:
  - Rz   : 6
  - Ry   : 3
  - Rx   : 1
  - CNOT : 1

=== fredkin_n3 ===
Circuit Statistics — fredkin_n3
-------------------------------
Qubits      : 3
Total gates : 23
Gate breakdown:
  - Rz   : 11
  - CNOT : 8
  - Rx   : 2
  - Ry   : 2

... (and so on for remaining circuits)
```

## Pipeline
The conversion flow follows these stages:
- **QASM**: Raw OpenQASM 2 source from QASMBench
- **AST**: `openqasm2-ir` parses the source into a validated abstract syntax tree (AST)
- **IR**: The AST lowers into the intermediate representation with normalized gates
- **GraphQOMB Circuit**: The IR is mapped onto GraphQOMB gate classes for downstream MBQC compilation

## License Note
This demo reads circuits directly from the QASMBench repository to respect the original license terms. No circuit files are copied or distributed with this demo.
