#!/usr/bin/env python
"""Standalone script for optimizing OpenQASM files using pytket.

This script provides a simple command-line interface for QASM optimization.
For more advanced usage, use the optimizer module directly.

Examples
--------
Basic usage::

    python tket_opt_qasm.py input.qasm output.qasm

Disable SWAP insertion::

    python tket_opt_qasm.py input.qasm output.qasm --no-swaps

Verbose output::

    python tket_opt_qasm.py input.qasm output.qasm -v
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

    Parameters
    ----------
    in_qasm_path : str or Path
        Path to the input QASM file.
    out_qasm_path : str or Path
        Path to the output (optimized) QASM file.
    allow_swaps : bool, optional
        Whether to allow SWAP gate insertion during optimization (default: True).
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
    circ = circuit_from_qasm(str(in_qasm_path))

    # Define optimization passes
    opt_pass = SequencePass(
        [
            FullPeepholeOptimise(allow_swaps=allow_swaps),
            RemoveRedundancies(),
        ]
    )

    # Apply optimization
    opt_pass.apply(circ)

    # pytket Circuit -> QASM file
    circuit_to_qasm(circ, str(out_qasm_path))


def main() -> int:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="tket を用いて OpenQASM ファイルを最適化します"
    )
    parser.add_argument("input_qasm", type=Path, help="入力 QASM ファイルパス")
    parser.add_argument("output_qasm", type=Path, help="出力（最適化後）QASM ファイルパス")
    parser.add_argument(
        "--no-swaps",
        action="store_true",
        help="FullPeepholeOptimise に allow_swaps=False を設定する場合に指定",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="詳細な出力を表示",
    )

    args = parser.parse_args()

    try:
        if args.verbose:
            print(f"入力ファイル: {args.input_qasm}")
            print(f"出力ファイル: {args.output_qasm}")
            print(f"SWAP許可: {not args.no_swaps}")

        optimise_qasm_with_tket(
            args.input_qasm,
            args.output_qasm,
            allow_swaps=not args.no_swaps,
        )

        if args.verbose:
            print(f"最適化完了: {args.output_qasm}")

        return 0

    except ImportError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        print("\npytket をインストールするには: pip install pytket", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"エラー: ファイルが見つかりません - {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
