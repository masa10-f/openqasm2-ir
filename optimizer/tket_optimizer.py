#!/usr/bin/env python
"""QASM optimizer using pytket (tket).

This module provides functions to optimize OpenQASM 2.0 circuits using pytket's
optimization passes. It supports both programmatic usage and command-line execution,
with options to output to files or return as strings.

Examples
--------
File to file optimization::

    from optimizer.tket_optimizer import optimise_qasm_with_tket

    optimise_qasm_with_tket(
        "input.qasm",
        "output_optimized.qasm",
        allow_swaps=True
    )

File to string optimization::

    from optimizer.tket_optimizer import optimise_qasm_to_string

    optimized_qasm = optimise_qasm_to_string("input.qasm")
    # Use the string directly with other tools

String to string optimization::

    from optimizer.tket_optimizer import optimise_qasm_string

    qasm_code = "OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\n..."
    optimized = optimise_qasm_string(qasm_code)

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
import tempfile
from io import StringIO
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


def optimise_qasm_to_string(
    in_qasm_path: str | Path,
    *,
    allow_swaps: bool = True,
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
    allow_swaps : bool, optional
        Whether to allow SWAP gate insertion during optimization.
        Default is True.

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
        If the QASM file cannot be parsed.

    Examples
    --------
    Optimize a QASM file and get the result as a string::

        >>> from optimizer import optimise_qasm_to_string
        >>> optimized_qasm = optimise_qasm_to_string("bell_pair.qasm")
        >>> print(optimized_qasm)
        OPENQASM 2.0;
        include "qelib1.inc";
        ...

    Use with GraphQOMB or other tools::

        >>> optimized = optimise_qasm_to_string("circuit.qasm", allow_swaps=False)
        >>> # Convert to GraphQOMB or process further
        >>> # from qasm2.parser import parse_qasm
        >>> # ast = parse_qasm(optimized)

    Notes
    -----
    This function uses the same optimization passes as `optimise_qasm_with_tket`:
    FullPeepholeOptimise and RemoveRedundancies.

    See Also
    --------
    optimise_qasm_with_tket : Optimize QASM file and write to another file
    optimise_qasm_string : Optimize a QASM string directly
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

    # QASM -> pytket Circuit
    try:
        circ = circuit_from_qasm(str(in_qasm_path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input QASM file not found: {in_qasm_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM file '{in_qasm_path}': {exc}") from exc

    # Define and apply optimization passes
    opt_pass = SequencePass(
        [
            FullPeepholeOptimise(allow_swaps=allow_swaps),
            RemoveRedundancies(),
        ]
    )
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
    allow_swaps: bool = True,
) -> str:
    """Optimize a QASM string using tket and return the optimized result.

    This function takes an OpenQASM 2.0 string, applies pytket's optimization passes,
    and returns the optimized circuit as a QASM string. This is the most flexible
    function for in-memory QASM processing.

    Parameters
    ----------
    qasm_string : str
        The input QASM code as a string.
    allow_swaps : bool, optional
        Whether to allow SWAP gate insertion during optimization.
        Default is True.

    Returns
    -------
    str
        The optimized QASM code as a string.

    Raises
    ------
    ImportError
        If pytket is not installed.
    ValueError
        If the QASM string cannot be parsed.

    Examples
    --------
    Optimize a QASM string::

        >>> from optimizer import optimise_qasm_string
        >>> qasm = '''OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q[0] -> c[0];
        ... measure q[1] -> c[1];
        ... '''
        >>> optimized = optimise_qasm_string(qasm)
        >>> print(optimized)

    Use in a pipeline::

        >>> qasm_code = read_qasm_from_somewhere()
        >>> optimized = optimise_qasm_string(qasm_code, allow_swaps=False)
        >>> process_with_graphqomb(optimized)

    Notes
    -----
    This function uses the same optimization passes as `optimise_qasm_with_tket`:
    FullPeepholeOptimise and RemoveRedundancies.

    See Also
    --------
    optimise_qasm_with_tket : Optimize QASM file and write to another file
    optimise_qasm_to_string : Optimize a QASM file and return as string
    """
    try:
        from pytket.passes import FullPeepholeOptimise, RemoveRedundancies, SequencePass
        from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm
    except ImportError as exc:
        raise ImportError(
            "pytket is required for QASM optimization. "
            "Install it with: pip install pytket"
        ) from exc

    # QASM string -> pytket Circuit
    try:
        circ = circuit_from_qasm_str(qasm_string)
    except Exception as exc:
        raise ValueError(f"Failed to parse QASM string: {exc}") from exc

    # Define and apply optimization passes
    opt_pass = SequencePass(
        [
            FullPeepholeOptimise(allow_swaps=allow_swaps),
            RemoveRedundancies(),
        ]
    )
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
