from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources as importlib_resources
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ir.circuit import Circuit as IRCircuit
from qasm2.errors import QasmError
from qasm2.lower import load_gate_mappings, lower_to_ir
from qasm2.normalize import normalize_program
from qasm2.parser import parse_qasm_file
from qasm2.validate import validate_program

_LOG = logging.getLogger("qasm-import")


@dataclass
class GraphQOMBArtifact:
    """Container bundling GraphQOMB circuit data with measurement metadata."""

    circuit: Any
    meas_map: list[tuple[int, int]] | None


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    examples = (
        "Examples:\n"
        "  qasm-import --in circuit.qasm --out circuit.json\n"
        "  qasm-import --in circuit.qasm --out circuit.pkl --format graphqomb"
    )
    parser = argparse.ArgumentParser(
        prog="qasm-import",
        description="Import an OpenQASM 2 circuit and lower it to the custom IR or GraphQOMB.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--in",
        dest="input_path",
        required=True,
        help="Path to the OpenQASM 2 source file that should be converted.",
    )
    parser.add_argument(
        "-o",
        "--out",
        dest="output_path",
        required=True,
        help="Destination path for the converted output.",
    )
    parser.add_argument(
        "-g",
        "--gates",
        dest="gates_path",
        help="Path to the gates.yaml mapping file (defaults to the packaged resource).",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["json", "graphqomb"],
        default="json",
        help="Output representation to generate (default: json).",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Enable strict validation of the input program (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict validation checks.",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Emit verbose logging (debug level).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_distribution_version()}",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure the logging subsystem for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _distribution_version() -> str:
    """Best-effort lookup of the installed package version."""
    try:
        return importlib.metadata.version("qasm2graphqomb")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _default_gates_path() -> str:
    """Locate the packaged gates.yaml resource."""
    candidates = ("qasm2", "qasm2.gates", "gates")
    for package in candidates:
        try:
            root = importlib_resources.files(package)
        except (ModuleNotFoundError, AttributeError):
            continue
        candidate = root.joinpath("gates.yaml")
        if candidate.is_file():
            with importlib_resources.as_file(candidate) as resolved:
                return str(resolved)
    local_fallback = Path(__file__).resolve().parent.parent / "gates" / "gates.yaml"
    if local_fallback.exists():
        return str(local_fallback)
    raise FileNotFoundError("Unable to locate packaged gates.yaml; please provide --gates explicitly.")


def _report_qasm_error(err: QasmError) -> None:
    """Print a formatted QASM diagnostic to stderr."""
    print(f"Error {err.code} at line {err.line}, col {err.col}: {err.message}", file=sys.stderr)


def _convert_to_graphqomb(ir_circuit: IRCircuit) -> GraphQOMBArtifact:
    """Convert the intermediate representation to a GraphQOMB circuit.

    This function delegates to the converter module which dynamically loads
    gate builders from gates.yaml, providing a single source of truth for
    gate conversion logic.
    """
    try:
        from graphqomb_adapter.converter import ir_to_graphqomb
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError("GraphQOMB must be installed to emit graphqomb format.") from exc

    graph_circuit = ir_to_graphqomb(ir_circuit)
    meas_map = list(ir_circuit.meas_map) if ir_circuit.meas_map is not None else None
    return GraphQOMBArtifact(circuit=graph_circuit, meas_map=meas_map)


def _write_json(circuit: IRCircuit, destination: str) -> None:
    """Persist the IR circuit as JSON."""
    circuit.to_json(destination)


def _write_graphqomb(artifact: GraphQOMBArtifact, destination: str) -> None:
    """Persist the GraphQOMB artifact as a pickle payload."""
    with Path(destination).open("wb") as handle:
        pickle.dump(artifact, handle)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    input_path = args.input_path
    output_path = args.output_path
    output_format = args.output_format

    _LOG.debug("Input file: %s", input_path)
    _LOG.debug("Output file: %s", output_path)
    _LOG.debug("Requested format: %s", output_format)

    try:
        gates_path = args.gates_path or _default_gates_path()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _LOG.debug("Using gate mappings: %s", gates_path)

    try:
        ast = parse_qasm_file(input_path)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"Failed to read input file '{input_path}': {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error reading input file '{input_path}': {exc}", file=sys.stderr)
        return 1
    except QasmError as err:
        _report_qasm_error(err)
        return 1

    try:
        if args.strict:
            _LOG.debug("Running strict validation.")
            validate_program(ast)
        else:
            _LOG.debug("Strict validation disabled by user.")
        normalized = normalize_program(ast)
        _LOG.debug("Normalization complete.")
    except QasmError as err:
        _report_qasm_error(err)
        return 1

    try:
        gate_mappings = load_gate_mappings(gates_path)
        _LOG.debug("Loaded gate mappings.")
    except FileNotFoundError as exc:
        print(f"Gate mapping file not found '{gates_path}': {exc}", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"Insufficient permissions to read gate mapping '{gates_path}': {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Invalid gate mapping file '{gates_path}': {exc}", file=sys.stderr)
        return 1

    try:
        circuit = lower_to_ir(normalized, gate_mappings)
        _LOG.debug("Lowering to IR complete.")
    except QasmError as err:
        _report_qasm_error(err)
        return 1

    try:
        if output_format == "json":
            _write_json(circuit, output_path)
        else:
            artifact = _convert_to_graphqomb(circuit)
            _write_graphqomb(artifact, output_path)
    except ImportError as exc:
        print(f"GraphQOMB conversion unavailable: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Failed to convert circuit: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Failed to write output file '{output_path}': {exc}", file=sys.stderr)
        return 1

    _LOG.info("Converted %d gates, %d qubits", len(circuit.ops), circuit.n_qubits)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
