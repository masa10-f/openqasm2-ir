"""QASM optimization module using pytket.

This module provides tools for optimizing OpenQASM 2.0 quantum circuits
using pytket's optimization passes. It supports file-to-file, file-to-string,
and string-to-string optimization workflows with configurable optimization levels.

Optimization Levels
-------------------
The module provides 5 optimization levels (0-4):

- **Level 0 (MINIMAL)**: RemoveRedundancies only (fastest)
- **Level 1 (CLIFFORD)**: CliffordSimp + RemoveRedundancies
- **Level 2 (PEEPHOLE_2Q)**: PeepholeOptimise2Q + RemoveRedundancies (default)
- **Level 3 (FULL_NO_SWAPS)**: FullPeepholeOptimise without SWAP insertion
- **Level 4 (FULL_WITH_SWAPS)**: FullPeepholeOptimise with SWAP insertion

Functions
---------
optimise_qasm_with_tket
    Optimize a QASM file using pytket and write the result to a new file.
optimise_qasm_to_string
    Optimize a QASM file using pytket and return the result as a string.
optimise_qasm_string
    Optimize a QASM string using pytket and return the optimized result.

Classes
-------
OptimizationLevel
    Enum for specifying optimization levels.

Examples
--------
File to file optimization with default level (2)::

    from optimizer import optimise_qasm_with_tket

    optimise_qasm_with_tket("input.qasm", "output_optimized.qasm")

File to string optimization with custom level::

    from optimizer import optimise_qasm_to_string, OptimizationLevel

    optimized_qasm = optimise_qasm_to_string(
        "input.qasm",
        optimization_level=OptimizationLevel.FULL_NO_SWAPS
    )

String to string optimization::

    from optimizer import optimise_qasm_string

    qasm_code = read_qasm_from_somewhere()
    optimized = optimise_qasm_string(qasm_code, optimization_level=0)  # fastest
"""

from optimizer.tket_optimizer import (
    OptimizationLevel,
    optimise_qasm_string,
    optimise_qasm_to_string,
    optimise_qasm_with_tket,
)

__all__ = [
    "OptimizationLevel",
    "optimise_qasm_with_tket",
    "optimise_qasm_to_string",
    "optimise_qasm_string",
]
