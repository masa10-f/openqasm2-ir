#!/usr/bin/env python
"""QASM optimizer using pytket (tket).

This module provides functions to optimize OpenQASM 2.0 circuits using pytket's
optimization passes. It supports both programmatic usage and command-line execution.

Examples
--------
As a module::

    from optimizer.tket_optimizer import optimise_qasm_with_tket

    optimise_qasm_with_tket(
        "input.qasm",
        "output_optimized.qasm",
        allow_swaps=True
    )

As a command-line tool::

    python -m optimizer.tket_optimizer input.qasm output.qasm
    python -m optimizer.tket_optimizer input.qasm output.qasm --no-swaps

Notes
-----
This optimizer uses pytket's `FullPeepholeOptimise` and `RemoveRedundancies` passes
to reduce gate count and circuit depth. The optimization is particularly effective
for Clifford circuits and small qubit patterns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def optimise_qasm_with_tket(
    in_qasm_path: str | Path,
    out_qasm_path: str | Path,
    *,
    allow_swaps: bool = True,
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
    allow_swaps : bool, optional
        Whether to allow SWAP gate insertion during optimization.
        Default is True. When False, the optimizer will not introduce
        additional SWAP gates, which may be useful for maintaining
        specific qubit mappings.

    Raises
    ------
    ImportError
        If pytket is not installed.
    FileNotFoundError
        If the input QASM file does not exist.
    ValueError
        If the QASM file cannot be parsed.

    Examples
    --------
    Optimize a QASM file with default settings::

        >>> optimise_qasm_with_tket("bell_pair.qasm", "bell_pair_opt.qasm")

    Optimize without allowing SWAP insertion::

        >>> optimise_qasm_with_tket(
        ...     "circuit.qasm",
        ...     "circuit_opt.qasm",
        ...     allow_swaps=False
        ... )

    Notes
    -----
    The optimization applies the following passes in sequence:

    1. **FullPeepholeOptimise**: A comprehensive optimization pass that includes:
       - Clifford simplification
       - Pattern matching for 1-3 qubit gate sequences
       - Rotation gate merging
       - Commutation-based optimizations

    2. **RemoveRedundancies**: Removes:
       - Gate-inverse pairs
       - Zero-angle rotation gates
       - Other redundant operations

    See Also
    --------
    pytket.passes.FullPeepholeOptimise : Comprehensive circuit optimization
    pytket.passes.RemoveRedundancies : Remove redundant gates
    """
    try:
        from pytket.passes import FullPeepholeOptimise, RemoveRedundancies, SequencePass
        from pytket.qasm import circuit_from_qasm, circuit_to_qasm
    except ImportError as exc:
        raise ImportError(
            "pytket is required for QASM optimization. "
            "Install it with: pip install pytket"
        ) from exc

    in_qasm_path = Path(in_qasm_path)
    out_qasm_path = Path(out_qasm_path)

    # QASM -> pytket Circuit
    try:
        circ = circuit_from_qasm(str(in_qasm_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input QASM file not found: {in_qasm_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM file '{in_qasm_path}': {exc}") from exc

    # Define optimization passes
    # FullPeepholeOptimise: Comprehensive optimization including Clifford reduction
    #                       and pattern matching for 1-3 qubit gates
    # RemoveRedundancies: Remove gate-inverse pairs and zero-angle gates
    opt_pass = SequencePass(
        [
            FullPeepholeOptimise(allow_swaps=allow_swaps),
            RemoveRedundancies(),
        ]
    )

    # Apply optimization to the circuit
    opt_pass.apply(circ)

    # pytket Circuit -> QASM file
    try:
        circuit_to_qasm(circ, str(out_qasm_path))
    except Exception as exc:
        raise IOError(f"Failed to write optimized QASM to '{out_qasm_path}': {exc}") from exc


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
            "Examples:\n"
            "  python -m optimizer.tket_optimizer input.qasm output.qasm\n"
            "  python -m optimizer.tket_optimizer circuit.qasm opt.qasm --no-swaps"
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
        "--no-swaps",
        action="store_true",
        help="Disable SWAP gate insertion (sets allow_swaps=False)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    try:
        if args.verbose:
            print(f"Input file: {args.input_qasm}")
            print(f"Output file: {args.output_qasm}")
            print(f"Allow swaps: {not args.no_swaps}")

        optimise_qasm_with_tket(
            args.input_qasm,
            args.output_qasm,
            allow_swaps=not args.no_swaps,
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
