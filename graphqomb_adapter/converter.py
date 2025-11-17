"""Converters between the intermediate circuit IR and GraphQOMB circuits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml
from graphqomb.circuit import Circuit
from graphqomb import gates

from ir.circuit import Circuit as IRCircuit, Op
from qasm2.ast_nodes import ProgramAST
from qasm2.lower import load_gate_mappings, lower_to_ir
from qasm2.normalize import normalize_program
from qasm2.parser import parse_qasm, parse_qasm_file
from qasm2.validate import validate_program

__all__ = ["ir_to_graphqomb", "qasm_to_graphqomb", "qasm_file_to_graphqomb"]

GateFactory = Callable[[Op], gates.Gate]


def _require_qubits(op: Op, expected: int) -> tuple[int, ...]:
    """Validate the number of qubit operands for an operation."""
    actual = len(op.qubits)
    if actual != expected:
        plural = "s" if expected != 1 else ""
        raise ValueError(f"IR gate '{op.name}' expects {expected} qubit operand{plural} but received {actual}.")
    return tuple(op.qubits)


def _require_params(op: Op, expected: int) -> tuple[float, ...]:
    """Validate the number of parameters for an operation."""
    actual = len(op.params)
    if actual != expected:
        plural = "s" if expected != 1 else ""
        raise ValueError(f"IR gate '{op.name}' expects {expected} parameter{plural} but received {actual}.")
    return tuple(op.params)


# Mapping from gate names to GraphQOMB gate classes
# Some gate names in IR differ from GraphQOMB class names
_GATE_CLASS_MAP = {
    "RX": gates.Rx,
    "RY": gates.Ry,
    "RZ": gates.Rz,
    "X": gates.X,
    "Y": gates.Y,
    "Z": gates.Z,
    "H": gates.H,
    "S": gates.S,
    "T": gates.T,
    "TDG": gates.Tdg,
    "U3": gates.U3,
    "CX": gates.CNOT,  # CX maps to CNOT
    "CZ": gates.CZ,
    "SWAP": gates.SWAP,
    "CRZ": gates.CRz,
    "CRX": gates.CRx,
    "CU3": gates.CU3,
    "CCX": gates.Toffoli,  # CCX maps to Toffoli
    "CCZ": gates.CCZ,
}


def _load_gate_specs(yaml_path: str | None = None) -> dict:
    """Load gate specifications from gates.yaml.

    Parameters
    ----------
    yaml_path : str | None
        Path to gates.yaml file. If None, uses default path.

    Returns
    -------
    dict
        Dictionary of gate specifications with arity and params.
    """
    if yaml_path is None:
        # Default path relative to this module
        current_dir = Path(__file__).parent
        yaml_path = current_dir.parent / "gates" / "gates.yaml"

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    return data.get("primitives", {})


def _create_gate_builder(gate_name: str, gate_spec: dict, gate_class: type) -> GateFactory:
    """Create a gate builder function from YAML specification.

    Parameters
    ----------
    gate_name : str
        Name of the gate (e.g., "RX", "H").
    gate_spec : dict
        Gate specification from YAML with 'arity' and 'params' keys.
    gate_class : type
        GraphQOMB gate class to instantiate.

    Returns
    -------
    GateFactory
        Factory function that creates gate instances from IR operations.
    """
    arity = gate_spec["arity"]
    param_names = gate_spec.get("params", [])
    num_params = len(param_names)

    def builder(op: Op) -> gates.Gate:
        qubits = _require_qubits(op, arity)
        params = _require_params(op, num_params)

        # Build gate based on arity and parameter count
        if arity == 1:
            qubit = qubits[0]
            if num_params == 0:
                return gate_class(qubit=qubit)
            elif num_params == 1:
                return gate_class(qubit=qubit, angle=params[0])
            elif num_params == 3:  # U3 gate
                return gate_class(qubit=qubit, angle1=params[0], angle2=params[1], angle3=params[2])
        elif arity == 2:
            if num_params == 0:
                return gate_class(qubits=qubits)
            elif num_params == 1:
                return gate_class(qubits=qubits, angle=params[0])
            elif num_params == 3:  # CU3 gate
                return gate_class(qubits=qubits, angle1=params[0], angle2=params[1], angle3=params[2])
        elif arity == 3:
            return gate_class(qubits=list(qubits))

        raise ValueError(f"Unsupported gate configuration: {gate_name} with arity={arity}, params={num_params}")

    return builder


def _load_gate_builders(yaml_path: str | None = None) -> dict[str, GateFactory]:
    """Load gate builders dynamically from gates.yaml.

    Parameters
    ----------
    yaml_path : str | None
        Path to gates.yaml file. If None, uses default path.

    Returns
    -------
    dict[str, GateFactory]
        Dictionary mapping gate names to builder functions.
    """
    gate_specs = _load_gate_specs(yaml_path)
    builders = {}

    for gate_name, gate_spec in gate_specs.items():
        if gate_name in _GATE_CLASS_MAP:
            gate_class = _GATE_CLASS_MAP[gate_name]
            builders[gate_name] = _create_gate_builder(gate_name, gate_spec, gate_class)

    return builders


# Load gate builders dynamically from gates.yaml at module initialization
# This serves as the default when no custom gates.yaml is provided
_DEFAULT_GATE_BUILDERS: dict[str, GateFactory] = _load_gate_builders()


def ir_to_graphqomb(ir_circuit: IRCircuit, gates_yaml_path: str | None = None) -> Circuit:
    """Convert an intermediate representation circuit into a GraphQOMB circuit.

    Parameters
    ----------
    ir_circuit : ir.circuit.Circuit
        Intermediate representation circuit to convert.
    gates_yaml_path : str | None, optional
        Path to the gate mapping YAML file. If None, uses the default packaged gates.yaml.
        This allows custom gate definitions to be honored during GraphQOMB conversion.

    Returns
    -------
    graphqomb.circuit.Circuit
        GraphQOMB circuit containing the mapped macro gates.

    Raises
    ------
    ValueError
        If the IR contains an unsupported gate or inconsistent operands.
    """
    # Load builders from custom YAML if provided, otherwise use default
    if gates_yaml_path is not None:
        gate_builders = _load_gate_builders(gates_yaml_path)
    else:
        gate_builders = _DEFAULT_GATE_BUILDERS

    circuit = Circuit(ir_circuit.n_qubits)
    for op in ir_circuit.ops:
        op_name = op.name.upper()
        builder = gate_builders.get(op_name)
        if builder is None:
            raise ValueError(f"Unsupported IR gate '{op.name}'.")
        gate = builder(op)
        circuit.apply_macro_gate(gate)
    return circuit


def _ast_to_graphqomb(ast: ProgramAST, gates_yaml_path: str) -> Circuit:
    """Convert an OpenQASM AST to a GraphQOMB circuit.

    Uses the same gates.yaml for both IR lowering and GraphQOMB conversion,
    ensuring custom gate definitions are honored throughout the pipeline.
    """
    validate_program(ast)
    normalized_ast = normalize_program(ast)
    gate_mappings = load_gate_mappings(gates_yaml_path)
    ir_circuit = lower_to_ir(normalized_ast, gate_mappings)
    return ir_to_graphqomb(ir_circuit, gates_yaml_path=gates_yaml_path)


def qasm_to_graphqomb(qasm_text: str, gates_yaml_path: str) -> Circuit:
    """Convert OpenQASM 2 source code into a GraphQOMB circuit.

    Parameters
    ----------
    qasm_text : str
        OpenQASM 2 program text.
    gates_yaml_path : str
        Path to the gate mapping YAML file used during lowering.

    Returns
    -------
    graphqomb.circuit.Circuit
        GraphQOMB circuit constructed from the supplied source.
    """
    ast = parse_qasm(qasm_text)
    return _ast_to_graphqomb(ast, gates_yaml_path)


def qasm_file_to_graphqomb(qasm_file_path: str, gates_yaml_path: str) -> Circuit:
    """Convert an OpenQASM 2 file into a GraphQOMB circuit.

    Parameters
    ----------
    qasm_file_path : str
        Path to a file containing OpenQASM 2 source.
    gates_yaml_path : str
        Path to the gate mapping YAML file used during lowering.

    Returns
    -------
    graphqomb.circuit.Circuit
        GraphQOMB circuit constructed from the file contents.
    """
    ast = parse_qasm_file(qasm_file_path)
    return _ast_to_graphqomb(ast, gates_yaml_path)
