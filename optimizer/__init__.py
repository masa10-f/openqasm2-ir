"""QASM optimization module using pytket.

This module provides tools for optimizing OpenQASM 2.0 quantum circuits
using pytket's optimization passes.

Functions
---------
optimise_qasm_with_tket
    Optimize a QASM file using pytket and write the result to a new file.

Examples
--------
Basic usage::

    from optimizer import optimise_qasm_with_tket

    optimise_qasm_with_tket("input.qasm", "output_optimized.qasm")

With custom options::

    optimise_qasm_with_tket(
        "circuit.qasm",
        "circuit_opt.qasm",
        allow_swaps=False
    )
"""
from optimizer.tket_optimizer import optimise_qasm_with_tket

__all__ = ["optimise_qasm_with_tket"]
