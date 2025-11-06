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
from ir.circuit import Op as IROp
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


def _ensure_qubits(op: IROp, expected: int) -> None:
    if len(op.qubits) != expected:
        raise ValueError(f"Gate '{op.name}' expects {expected} qubits but received {len(op.qubits)}.")


def _ensure_params(op: IROp, expected: int) -> None:
    if len(op.params) != expected:
        raise ValueError(f"Gate '{op.name}' expects {expected} parameters but received {len(op.params)}.")


def _convert_to_graphqomb(ir_circuit: IRCircuit) -> GraphQOMBArtifact:
    """Convert the intermediate representation to a GraphQOMB circuit."""
    try:
        from graphqomb.circuit import Circuit as GraphCircuit
        from graphqomb.gates import CCZ as GraphCCZ
        from graphqomb.gates import CRx as GraphCRx
        from graphqomb.gates import CRz as GraphCRz
        from graphqomb.gates import CU3 as GraphCU3
        from graphqomb.gates import CNOT as GraphCNOT
        from graphqomb.gates import CZ as GraphCZ
        from graphqomb.gates import Rx as GraphRx
        from graphqomb.gates import Ry as GraphRy
        from graphqomb.gates import Rz as GraphRz
        from graphqomb.gates import SWAP as GraphSWAP
        from graphqomb.gates import Toffoli as GraphToffoli
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise ImportError("GraphQOMB must be installed to emit graphqomb format.") from exc

    graph_circuit = GraphCircuit(ir_circuit.n_qubits)

    for op in ir_circuit.ops:
        name = op.name.upper()
        if name == "RX":
            _ensure_qubits(op, 1)
            _ensure_params(op, 1)
            gate = GraphRx(qubit=op.qubits[0], angle=float(op.params[0]))
        elif name == "RY":
            _ensure_qubits(op, 1)
            _ensure_params(op, 1)
            gate = GraphRy(qubit=op.qubits[0], angle=float(op.params[0]))
        elif name == "RZ":
            _ensure_qubits(op, 1)
            _ensure_params(op, 1)
            gate = GraphRz(qubit=op.qubits[0], angle=float(op.params[0]))
        elif name == "CX":
            _ensure_qubits(op, 2)
            _ensure_params(op, 0)
            gate = GraphCNOT(qubits=(op.qubits[0], op.qubits[1]))
        elif name == "CZ":
            _ensure_qubits(op, 2)
            _ensure_params(op, 0)
            gate = GraphCZ(qubits=(op.qubits[0], op.qubits[1]))
        elif name == "SWAP":
            _ensure_qubits(op, 2)
            _ensure_params(op, 0)
            gate = GraphSWAP(qubits=(op.qubits[0], op.qubits[1]))
        elif name == "CRZ":
            _ensure_qubits(op, 2)
            _ensure_params(op, 1)
            gate = GraphCRz(qubits=(op.qubits[0], op.qubits[1]), angle=float(op.params[0]))
        elif name == "CRX":
            _ensure_qubits(op, 2)
            _ensure_params(op, 1)
            gate = GraphCRx(qubits=(op.qubits[0], op.qubits[1]), angle=float(op.params[0]))
        elif name == "CU3":
            _ensure_qubits(op, 2)
            _ensure_params(op, 3)
            gate = GraphCU3(
                qubits=(op.qubits[0], op.qubits[1]),
                angle1=float(op.params[0]),
                angle2=float(op.params[1]),
                angle3=float(op.params[2]),
            )
        elif name == "CCX":
            _ensure_qubits(op, 3)
            _ensure_params(op, 0)
            gate = GraphToffoli(qubits=list(op.qubits))
        elif name == "CCZ":
            _ensure_qubits(op, 3)
            _ensure_params(op, 0)
            gate = GraphCCZ(qubits=list(op.qubits))
        else:  # pragma: no cover - depends on gate mappings
            raise ValueError(f"Unsupported gate '{op.name}' for GraphQOMB conversion.")
        graph_circuit.apply_macro_gate(gate)

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
