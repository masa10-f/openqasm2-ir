"""Converters between the intermediate circuit IR and GraphQOMB circuits."""

from __future__ import annotations

from collections.abc import Callable

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


def _build_rx(op: Op) -> gates.Rx:
    qubit = _require_qubits(op, 1)[0]
    angle = _require_params(op, 1)[0]
    return gates.Rx(qubit=qubit, angle=angle)


def _build_ry(op: Op) -> gates.Ry:
    qubit = _require_qubits(op, 1)[0]
    angle = _require_params(op, 1)[0]
    return gates.Ry(qubit=qubit, angle=angle)


def _build_rz(op: Op) -> gates.Rz:
    qubit = _require_qubits(op, 1)[0]
    angle = _require_params(op, 1)[0]
    return gates.Rz(qubit=qubit, angle=angle)


def _build_cx(op: Op) -> gates.CNOT:
    control, target = _require_qubits(op, 2)
    _require_params(op, 0)
    return gates.CNOT(qubits=(control, target))


def _build_cz(op: Op) -> gates.CZ:
    qubits = _require_qubits(op, 2)
    _require_params(op, 0)
    return gates.CZ(qubits=qubits)


def _build_swap(op: Op) -> gates.SWAP:
    qubits = _require_qubits(op, 2)
    _require_params(op, 0)
    return gates.SWAP(qubits=qubits)


def _build_crz(op: Op) -> gates.CRz:
    control, target = _require_qubits(op, 2)
    angle = _require_params(op, 1)[0]
    return gates.CRz(qubits=(control, target), angle=angle)


def _build_crx(op: Op) -> gates.CRx:
    control, target = _require_qubits(op, 2)
    angle = _require_params(op, 1)[0]
    return gates.CRx(qubits=(control, target), angle=angle)


def _build_cu3(op: Op) -> gates.CU3:
    qubits = _require_qubits(op, 2)
    theta, phi, lam = _require_params(op, 3)
    return gates.CU3(qubits=qubits, angle1=theta, angle2=phi, angle3=lam)


def _build_ccx(op: Op) -> gates.Toffoli:
    qubits = list(_require_qubits(op, 3))
    _require_params(op, 0)
    return gates.Toffoli(qubits=qubits)


def _build_ccz(op: Op) -> gates.CCZ:
    qubits = list(_require_qubits(op, 3))
    _require_params(op, 0)
    return gates.CCZ(qubits=qubits)


_GATE_BUILDERS: dict[str, GateFactory] = {
    "RX": _build_rx,
    "RY": _build_ry,
    "RZ": _build_rz,
    "CX": _build_cx,
    "CZ": _build_cz,
    "SWAP": _build_swap,
    "CRZ": _build_crz,
    "CRX": _build_crx,
    "CU3": _build_cu3,
    "CCX": _build_ccx,
    "CCZ": _build_ccz,
}


def ir_to_graphqomb(ir_circuit: IRCircuit) -> Circuit:
    """Convert an intermediate representation circuit into a GraphQOMB circuit.

    Parameters
    ----------
    ir_circuit : ir.circuit.Circuit
        Intermediate representation circuit to convert.

    Returns
    -------
    graphqomb.circuit.Circuit
        GraphQOMB circuit containing the mapped macro gates.

    Raises
    ------
    ValueError
        If the IR contains an unsupported gate or inconsistent operands.
    """
    circuit = Circuit(ir_circuit.n_qubits)
    for op in ir_circuit.ops:
        op_name = op.name.upper()
        builder = _GATE_BUILDERS.get(op_name)
        if builder is None:
            raise ValueError(f"Unsupported IR gate '{op.name}'.")
        gate = builder(op)
        circuit.apply_macro_gate(gate)
    return circuit


def _ast_to_graphqomb(ast: ProgramAST, gates_yaml_path: str) -> Circuit:
    """Convert an OpenQASM AST to a GraphQOMB circuit."""
    validate_program(ast)
    normalized_ast = normalize_program(ast)
    gate_mappings = load_gate_mappings(gates_yaml_path)
    ir_circuit = lower_to_ir(normalized_ast, gate_mappings)
    return ir_to_graphqomb(ir_circuit)


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
