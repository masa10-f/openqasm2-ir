"""QASM optimization module using pytket.

This module provides tools for optimizing OpenQASM 2.0 quantum circuits
using pytket's optimization passes. It supports file-to-file, file-to-string,
and string-to-string optimization workflows.

Functions
---------
optimise_qasm_with_tket
    Optimize a QASM file using pytket and write the result to a new file.
optimise_qasm_to_string
    Optimize a QASM file using pytket and return the result as a string.
optimise_qasm_string
    Optimize a QASM string using pytket and return the optimized result.

Examples
--------
File to file optimization::

    from optimizer import optimise_qasm_with_tket

    optimise_qasm_with_tket("input.qasm", "output_optimized.qasm")

File to string optimization (useful for GraphQOMB conversion)::

    from optimizer import optimise_qasm_to_string

    optimized_qasm = optimise_qasm_to_string("input.qasm")
    # Process the string directly with other tools

String to string optimization::

    from optimizer import optimise_qasm_string

    qasm_code = read_qasm_from_somewhere()
    optimized = optimise_qasm_string(qasm_code, allow_swaps=False)

With custom options::

    optimise_qasm_with_tket(
        "circuit.qasm",
        "circuit_opt.qasm",
        allow_swaps=False
    )
"""
from optimizer.tket_optimizer import (
    optimise_qasm_string,
    optimise_qasm_to_string,
    optimise_qasm_with_tket,
)

__all__ = [
    "optimise_qasm_with_tket",
    "optimise_qasm_to_string",
    "optimise_qasm_string",
]
