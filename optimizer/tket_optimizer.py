#!/usr/bin/env python
"""QASM optimizer using pytket (tket).

This module provides functions to optimize OpenQASM 2.0 circuits using pytket's
optimization passes. It supports both programmatic usage and command-line execution,
with options to output to files or return as strings.

Optimization Levels
-------------------
The module provides 5 optimization levels (0-4):

- **Level 0**: RemoveRedundancies only (fastest, minimal optimization)
- **Level 1**: CliffordSimp + RemoveRedundancies (Clifford gate simplification)
- **Level 2**: PeepholeOptimise2Q + RemoveRedundancies (default, good balance)
- **Level 3**: FullPeepholeOptimise without SWAP insertion + RemoveRedundancies
- **Level 4**: FullPeepholeOptimise with SWAP insertion + RemoveRedundancies (most aggressive)

Examples
--------
File to file optimization with optimization level::

    from optimizer import optimise_qasm_with_tket, OptimizationLevel

    optimise_qasm_with_tket(
        "input.qasm",
        "output_optimized.qasm",
        optimization_level=OptimizationLevel.PEEPHOLE_2Q
    )

String to string optimization::

    from optimizer import optimise_qasm_string

    optimized = optimise_qasm_string(qasm_code, optimization_level=2)

As a command-line tool::

    tket-optimize-qasm input.qasm output.qasm --level 2
    tket-optimize-qasm input.qasm output.qasm --level 4  # most aggressive

Notes
-----
This optimizer uses pytket's optimization passes to reduce gate count and circuit
depth. The optimization is particularly effective for Clifford circuits and small
qubit patterns.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket.passes import BasePass


class OptimizationLevel(IntEnum):
    """Optimization level for QASM circuit optimization.

    Attributes
    ----------
    MINIMAL : int
        Level 0: RemoveRedundancies only. Fastest optimization that removes
        gate-inverse pairs and zero-angle rotations.
    CLIFFORD : int
        Level 1: CliffordSimp + RemoveRedundancies. Simplifies Clifford gates
        while removing redundancies.
    PEEPHOLE_2Q : int
        Level 2: PeepholeOptimise2Q + RemoveRedundancies. Default level with
        good balance between optimization quality and speed.
    FULL_NO_SWAPS : int
        Level 3: FullPeepholeOptimise(allow_swaps=False) + RemoveRedundancies.
        Comprehensive optimization without SWAP gate insertion.
    FULL_WITH_SWAPS : int
        Level 4: FullPeepholeOptimise(allow_swaps=True) + RemoveRedundancies.
        Most aggressive optimization that may insert SWAP gates.

    Examples
    --------
    >>> from optimizer import OptimizationLevel
    >>> level = OptimizationLevel.PEEPHOLE_2Q
    >>> print(level.value)
    2
    >>> print(level.description)
    'PeepholeOptimise2Q + RemoveRedundancies (default, balanced)'
    """

    MINIMAL = 0
    CLIFFORD = 1
    PEEPHOLE_2Q = 2
    FULL_NO_SWAPS = 3
    FULL_WITH_SWAPS = 4

    @property
    def description(self) -> str:
        """Return a human-readable description of this optimization level."""
        descriptions = {
            0: "RemoveRedundancies only (fastest, minimal)",
            1: "CliffordSimp + RemoveRedundancies",
            2: "PeepholeOptimise2Q + RemoveRedundancies (default, balanced)",
            3: "FullPeepholeOptimise (no swaps) + RemoveRedundancies",
            4: "FullPeepholeOptimise (with swaps) + RemoveRedundancies (most aggressive)",
        }
        return descriptions[self.value]


def _get_optimization_pass(level: int | OptimizationLevel) -> "BasePass":
    """Get the pytket optimization pass for the given level.

    Parameters
    ----------
    level : int or OptimizationLevel
        The optimization level (0-4).

    Returns
    -------
    BasePass
        A pytket optimization pass sequence.

    Raises
    ------
    ImportError
        If pytket is not installed.
    ValueError
        If the level is not in the valid range (0-4).
    """
    try:
        from pytket.passes import (
            CliffordSimp,
            FullPeepholeOptimise,
            PeepholeOptimise2Q,
            RemoveRedundancies,
            SequencePass,
        )
    except ImportError as exc:
        raise ImportError("pytket is required for QASM optimization. Install it with: pip install pytket") from exc

    level_int = int(level)
    if level_int < 0 or level_int > 4:
        raise ValueError(f"Invalid optimization level: {level_int}. Must be 0-4.")

    if level_int == 0:
        return SequencePass([RemoveRedundancies()])
    elif level_int == 1:
        return SequencePass([CliffordSimp(), RemoveRedundancies()])
    elif level_int == 2:
        return SequencePass([PeepholeOptimise2Q(), RemoveRedundancies()])
    elif level_int == 3:
        return SequencePass([FullPeepholeOptimise(allow_swaps=False), RemoveRedundancies()])
    else:  # level_int == 4
        return SequencePass([FullPeepholeOptimise(allow_swaps=True), RemoveRedundancies()])


def optimise_qasm_with_tket(
    in_qasm_path: str | Path,
    out_qasm_path: str | Path,
    *,
    optimization_level: int | OptimizationLevel = OptimizationLevel.PEEPHOLE_2Q,
    allow_swaps: bool | None = None,
) -> None:
    """Optimize an OpenQASM file using tket and write the result to another file.

    This function reads an OpenQASM 2.0 file, applies pytket's optimization passes,
    and writes the optimized circuit to a new QASM file.

    Parameters
    ----------
    in_qasm_path : str or Path
        Path to the input QASM file.
    out_qasm_path : str or Path
        Path to the output (optimized) QASM file.
    optimization_level : int or OptimizationLevel, optional
        The optimization level to use (0-4). Default is 2 (PEEPHOLE_2Q).

        - 0: RemoveRedundancies only (fastest)
        - 1: CliffordSimp + RemoveRedundancies
        - 2: PeepholeOptimise2Q + RemoveRedundancies (default)
        - 3: FullPeepholeOptimise without SWAP insertion
        - 4: FullPeepholeOptimise with SWAP insertion (most aggressive)

    allow_swaps : bool, optional
        Deprecated. Use optimization_level instead.
        If provided, overrides optimization_level:
        - allow_swaps=True sets level to 4
        - allow_swaps=False sets level to 3

    Raises
    ------
    ImportError
        If pytket is not installed.
    FileNotFoundError
        If the input QASM file does not exist.
    ValueError
        If the QASM file cannot be parsed or optimization_level is invalid.

    Examples
    --------
    Optimize a QASM file with default settings (level 2)::

        >>> optimise_qasm_with_tket("bell_pair.qasm", "bell_pair_opt.qasm")

    Use minimal optimization for speed::

        >>> from optimizer import OptimizationLevel
        >>> optimise_qasm_with_tket(
        ...     "circuit.qasm",
        ...     "circuit_opt.qasm",
        ...     optimization_level=OptimizationLevel.MINIMAL
        ... )

    Use most aggressive optimization::

        >>> optimise_qasm_with_tket(
        ...     "circuit.qasm",
        ...     "circuit_opt.qasm",
        ...     optimization_level=4
        ... )

    Notes
    -----
    The optimization passes applied depend on the level:

    - **Level 0 (MINIMAL)**: Only removes redundant gates
    - **Level 1 (CLIFFORD)**: Clifford simplification
    - **Level 2 (PEEPHOLE_2Q)**: 2-qubit peephole optimization (recommended)
    - **Level 3 (FULL_NO_SWAPS)**: Full peephole without SWAP insertion
    - **Level 4 (FULL_WITH_SWAPS)**: Full peephole with SWAP insertion

    See Also
    --------
    OptimizationLevel : Enum for optimization levels
    optimise_qasm_to_string : Optimize and return as string
    optimise_qasm_string : Optimize a QASM string
    """
    try:
        from pytket.qasm import circuit_from_qasm, circuit_to_qasm
    except ImportError as exc:
        raise ImportError("pytket is required for QASM optimization. Install it with: pip install pytket") from exc

    # Handle deprecated allow_swaps parameter
    if allow_swaps is not None:
        import warnings

        warnings.warn(
            "allow_swaps is deprecated. Use optimization_level instead. "
            "allow_swaps=True maps to level 4, allow_swaps=False maps to level 3.",
            DeprecationWarning,
            stacklevel=2,
        )
        optimization_level = OptimizationLevel.FULL_WITH_SWAPS if allow_swaps else OptimizationLevel.FULL_NO_SWAPS

    in_qasm_path = Path(in_qasm_path)
    out_qasm_path = Path(out_qasm_path)

    # QASM -> pytket Circuit
    try:
        circ = circuit_from_qasm(str(in_qasm_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input QASM file not found: {in_qasm_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM file '{in_qasm_path}': {exc}") from exc

    # Get and apply optimization pass
    opt_pass = _get_optimization_pass(optimization_level)
    opt_pass.apply(circ)

    # pytket Circuit -> QASM file
    try:
        circuit_to_qasm(circ, str(out_qasm_path))
    except Exception as exc:
        raise IOError(f"Failed to write optimized QASM to '{out_qasm_path}': {exc}") from exc


def optimise_qasm_to_string(
    in_qasm_path: str | Path,
    *,
    optimization_level: int | OptimizationLevel = OptimizationLevel.PEEPHOLE_2Q,
    allow_swaps: bool | None = None,
) -> str:
    """Optimize an OpenQASM file using tket and return the result as a string.

    This function reads an OpenQASM 2.0 file, applies pytket's optimization passes,
    and returns the optimized circuit as a QASM string. This is useful when you
    want to process the optimized QASM directly without writing to a file, for
    example when converting to GraphQOMB or other formats.

    Parameters
    ----------
    in_qasm_path : str or Path
        Path to the input QASM file.
    optimization_level : int or OptimizationLevel, optional
        The optimization level to use (0-4). Default is 2 (PEEPHOLE_2Q).
    allow_swaps : bool, optional
        Deprecated. Use optimization_level instead.

    Returns
    -------
    str
        The optimized QASM code as a string.

    Raises
    ------
    ImportError
        If pytket is not installed.
    FileNotFoundError
        If the input QASM file does not exist.
    ValueError
        If the QASM file cannot be parsed or optimization_level is invalid.

    Examples
    --------
    Optimize a QASM file and get the result as a string::

        >>> from optimizer import optimise_qasm_to_string
        >>> optimized_qasm = optimise_qasm_to_string("bell_pair.qasm")
        >>> print(optimized_qasm)

    Use with different optimization levels::

        >>> fast_opt = optimise_qasm_to_string("circuit.qasm", optimization_level=0)
        >>> aggressive_opt = optimise_qasm_to_string("circuit.qasm", optimization_level=4)

    See Also
    --------
    optimise_qasm_with_tket : Optimize QASM file and write to another file
    optimise_qasm_string : Optimize a QASM string directly
    """
    try:
        from pytket.qasm import circuit_from_qasm, circuit_to_qasm
    except ImportError as exc:
        raise ImportError("pytket is required for QASM optimization. Install it with: pip install pytket") from exc

    # Handle deprecated allow_swaps parameter
    if allow_swaps is not None:
        import warnings

        warnings.warn(
            "allow_swaps is deprecated. Use optimization_level instead. "
            "allow_swaps=True maps to level 4, allow_swaps=False maps to level 3.",
            DeprecationWarning,
            stacklevel=2,
        )
        optimization_level = OptimizationLevel.FULL_WITH_SWAPS if allow_swaps else OptimizationLevel.FULL_NO_SWAPS

    in_qasm_path = Path(in_qasm_path)

    # QASM -> pytket Circuit
    try:
        circ = circuit_from_qasm(str(in_qasm_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input QASM file not found: {in_qasm_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM file '{in_qasm_path}': {exc}") from exc

    # Get and apply optimization pass
    opt_pass = _get_optimization_pass(optimization_level)
    opt_pass.apply(circ)

    # pytket Circuit -> QASM string (using temporary file)
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".qasm", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        circuit_to_qasm(circ, tmp_path)

        with open(tmp_path, "r") as f:
            qasm_string = f.read()

        # Clean up temporary file
        Path(tmp_path).unlink()

        return qasm_string
    except Exception as exc:
        raise IOError(f"Failed to generate optimized QASM string: {exc}") from exc


def optimise_qasm_string(
    qasm_string: str,
    *,
    optimization_level: int | OptimizationLevel = OptimizationLevel.PEEPHOLE_2Q,
    allow_swaps: bool | None = None,
) -> str:
    """Optimize a QASM string using tket and return the optimized result.

    This function takes an OpenQASM 2.0 string, applies pytket's optimization passes,
    and returns the optimized circuit as a QASM string. This is the most flexible
    function for in-memory QASM processing.

    Parameters
    ----------
    qasm_string : str
        The input QASM code as a string.
    optimization_level : int or OptimizationLevel, optional
        The optimization level to use (0-4). Default is 2 (PEEPHOLE_2Q).
    allow_swaps : bool, optional
        Deprecated. Use optimization_level instead.

    Returns
    -------
    str
        The optimized QASM code as a string.

    Raises
    ------
    ImportError
        If pytket is not installed.
    ValueError
        If the QASM string cannot be parsed or optimization_level is invalid.

    Examples
    --------
    Optimize a QASM string with default settings::

        >>> from optimizer import optimise_qasm_string
        >>> qasm = '''OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... h q[0];
        ... x q[0];
        ... x q[0];
        ... cx q[0], q[1];
        ... '''
        >>> optimized = optimise_qasm_string(qasm)

    Use different optimization levels::

        >>> from optimizer import optimise_qasm_string, OptimizationLevel
        >>> # Fast optimization
        >>> fast = optimise_qasm_string(qasm, optimization_level=0)
        >>> # Most aggressive
        >>> aggressive = optimise_qasm_string(qasm, optimization_level=OptimizationLevel.FULL_WITH_SWAPS)

    See Also
    --------
    optimise_qasm_with_tket : Optimize QASM file and write to another file
    optimise_qasm_to_string : Optimize a QASM file and return as string
    """
    try:
        from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm
    except ImportError as exc:
        raise ImportError("pytket is required for QASM optimization. Install it with: pip install pytket") from exc

    # Handle deprecated allow_swaps parameter
    if allow_swaps is not None:
        import warnings

        warnings.warn(
            "allow_swaps is deprecated. Use optimization_level instead. "
            "allow_swaps=True maps to level 4, allow_swaps=False maps to level 3.",
            DeprecationWarning,
            stacklevel=2,
        )
        optimization_level = OptimizationLevel.FULL_WITH_SWAPS if allow_swaps else OptimizationLevel.FULL_NO_SWAPS

    # QASM string -> pytket Circuit
    try:
        circ = circuit_from_qasm_str(qasm_string)
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM string: {exc}") from exc

    # Get and apply optimization pass
    opt_pass = _get_optimization_pass(optimization_level)
    opt_pass.apply(circ)

    # pytket Circuit -> QASM string (using temporary file)
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".qasm", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        circuit_to_qasm(circ, tmp_path)

        with open(tmp_path, "r") as f:
            qasm_string_out = f.read()

        # Clean up temporary file
        Path(tmp_path).unlink()

        return qasm_string_out
    except Exception as exc:
        raise IOError(f"Failed to generate optimized QASM string: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for QASM optimization.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        prog="tket-optimize-qasm",
        description="Optimize OpenQASM files using pytket (tket)",
        epilog=(
            "Optimization Levels:\n"
            "  0: RemoveRedundancies only (fastest)\n"
            "  1: CliffordSimp + RemoveRedundancies\n"
            "  2: PeepholeOptimise2Q + RemoveRedundancies (default)\n"
            "  3: FullPeepholeOptimise without SWAP insertion\n"
            "  4: FullPeepholeOptimise with SWAP insertion (most aggressive)\n"
            "\n"
            "Examples:\n"
            "  tket-optimize-qasm input.qasm output.qasm\n"
            "  tket-optimize-qasm input.qasm output.qasm --level 0  # fastest\n"
            "  tket-optimize-qasm input.qasm output.qasm --level 4  # most aggressive"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_qasm",
        type=Path,
        help="Path to the input QASM file",
    )
    parser.add_argument(
        "output_qasm",
        type=Path,
        help="Path to the output (optimized) QASM file",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=2,
        help="Optimization level (0-4, default: 2)",
    )
    parser.add_argument(
        "--no-swaps",
        action="store_true",
        help="Deprecated: Use --level 3 instead. Sets optimization level to 3.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    # Handle deprecated --no-swaps
    optimization_level = args.level
    if args.no_swaps:
        import warnings

        warnings.warn(
            "--no-swaps is deprecated. Use --level 3 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        optimization_level = 3

    try:
        if args.verbose:
            print(f"Input file: {args.input_qasm}")
            print(f"Output file: {args.output_qasm}")
            print(f"Optimization level: {optimization_level} ({OptimizationLevel(optimization_level).description})")

        optimise_qasm_with_tket(
            args.input_qasm,
            args.output_qasm,
            optimization_level=optimization_level,
        )

        if args.verbose:
            print(f"Successfully optimized QASM file: {args.output_qasm}")

        return 0

    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nTo install pytket, run: pip install pytket",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, IOError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
