"""Strict profile validator for OpenQASM 2 AST nodes.

This module enforces the subset of OpenQASM permitted by the strict profile. It
rejects classical control flow, mid-circuit measurements, invalid register
usage, and impure gate definitions. Validation is performed on the typed AST
emitted by :mod:`qasm2.parser`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from qasm2.ast_nodes import BarrierAST, GateCallAST, GateDefAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmError

__all__ = ["validate_program"]

_STRICT_VERSION = "2.0"
_ALLOWED_INCLUDES = frozenset({"qelib1.inc"})


@dataclass(frozen=True)
class _GateSignature:
    """Expected shape of a gate call."""

    num_params: int
    num_qubits: int


_BUILTIN_SIGNATURES: Dict[str, _GateSignature] = {
    "u1": _GateSignature(1, 1),
    "u2": _GateSignature(2, 1),
    "u3": _GateSignature(3, 1),
    "p": _GateSignature(1, 1),
    "rx": _GateSignature(1, 1),
    "ry": _GateSignature(1, 1),
    "rz": _GateSignature(1, 1),
    "x": _GateSignature(0, 1),
    "y": _GateSignature(0, 1),
    "z": _GateSignature(0, 1),
    "h": _GateSignature(0, 1),
    "s": _GateSignature(0, 1),
    "sdg": _GateSignature(0, 1),
    "t": _GateSignature(0, 1),
    "tdg": _GateSignature(0, 1),
    "id": _GateSignature(0, 1),
    "sx": _GateSignature(0, 1),
    "sxdg": _GateSignature(0, 1),
    "swap": _GateSignature(0, 2),
    "cx": _GateSignature(0, 2),
    "cz": _GateSignature(0, 2),
    "ccx": _GateSignature(0, 3),
    "cswap": _GateSignature(0, 3),
}


def validate_program(ast: ProgramAST) -> None:
    """Validate the strict-profile invariants on an OpenQASM program AST.

    Parameters
    ----------
    ast
        Parsed OpenQASM program.

    Raises
    ------
    QasmError
        If the AST violates any strict-profile constraint.
    """
    _validate_version(ast)
    _validate_includes(ast)
    qregs = _build_register_map(ast.qregs, "quantum")
    cregs = _build_register_map(ast.cregs, "classical")
    _validate_gate_definitions(ast.gate_defs)
    _validate_program_body(ast.body, qregs, cregs, ast.gate_defs)


def _validate_version(ast: ProgramAST) -> None:
    if ast.version != _STRICT_VERSION:
        raise QasmError(
            "E301",
            f"OPENQASM version '{ast.version}' is not supported; use {_STRICT_VERSION}.",
            1,
            1,
        )


def _validate_includes(ast: ProgramAST) -> None:
    for include in ast.includes:
        if include not in _ALLOWED_INCLUDES:
            raise QasmError(
                "E301",
                f"include '{include}' is not permitted; only qelib1.inc is allowed.",
                1,
                1,
            )


def _build_register_map(
    regs: Sequence[tuple[str, int]],
    kind: str,
) -> Dict[str, int]:
    seen: Dict[str, int] = {}
    for name, size in regs:
        if name in seen:
            raise QasmError(
                "E401",
                f"Duplicate {kind} register '{name}' is not allowed.",
                1,
                1,
            )
        if size <= 0:
            raise QasmError(
                "E402",
                f"{kind.capitalize()} register '{name}' must have size greater than zero.",
                1,
                1,
            )
        seen[name] = size
    return seen


def _validate_gate_definitions(gate_defs: Mapping[str, GateDefAST]) -> None:
    _ensure_gate_recursion_free(gate_defs)
    for gate in gate_defs.values():
        for call in gate.body:
            _assert_gate_known(call, gate_defs)
            _assert_gate_signature(call, gate_defs)


def _ensure_gate_recursion_free(gate_defs: Mapping[str, GateDefAST]) -> None:
    graph: Dict[str, Sequence[GateCallAST]] = {name: definition.body for name, definition in gate_defs.items()}
    visiting: Dict[str, bool] = {}
    visited: Dict[str, bool] = {}

    def dfs(gate_name: str) -> None:
        visiting[gate_name] = True
        for call in graph.get(gate_name, ()):
            target = call.name
            if target not in gate_defs:
                continue
            if visiting.get(target):
                raise QasmError(
                    "E501",
                    f"Recursive gate definition involving '{target}' is not allowed; remove the recursive call.",
                    call.line,
                    call.col,
                )
            if not visited.get(target):
                dfs(target)
        visiting.pop(gate_name, None)
        visited[gate_name] = True

    for name in gate_defs:
        if not visited.get(name):
            dfs(name)


def _validate_program_body(
    body: Sequence[GateCallAST | MeasureAST | BarrierAST],
    qregs: Mapping[str, int],
    cregs: Mapping[str, int],
    gate_defs: Mapping[str, GateDefAST],
) -> None:
    measurement_seen = False
    for stmt in body:
        if isinstance(stmt, GateCallAST):
            _assert_gate_known(stmt, gate_defs)
            _assert_gate_signature(stmt, gate_defs)
            _validate_qargs(stmt.qargs, qregs)
            if measurement_seen:
                raise QasmError(
                    "E201",
                    "mid-circuit measurement is not allowed; move all measurements to the end.",
                    stmt.line,
                    stmt.col,
                )
        elif isinstance(stmt, BarrierAST):
            _validate_qargs(stmt.qargs, qregs)
            if measurement_seen:
                raise QasmError(
                    "E201",
                    "mid-circuit measurement is not allowed; move all measurements to the end.",
                    stmt.line,
                    stmt.col,
                )
        elif isinstance(stmt, MeasureAST):
            _validate_measurement(stmt, qregs, cregs)
            measurement_seen = True
        else:
            raise QasmError(
                "E101",
                "Forbidden construct encountered; strict profile disallows classical control and resets.",
                getattr(stmt, "line", 1),
                getattr(stmt, "col", 1),
            )


def _validate_qargs(qargs: Iterable[QRef], qregs: Mapping[str, int]) -> None:
    for qref in qargs:
        size = qregs.get(qref.reg)
        if size is None:
            raise QasmError(
                "E402",
                f"Quantum register '{qref.reg}' is not defined.",
                qref.line,
                qref.col,
            )
        if qref.idx is not None and (qref.idx < 0 or qref.idx >= size):
            raise QasmError(
                "E402",
                f"Qubit index {qref.idx} is out of range for register '{qref.reg}'.",
                qref.line,
                qref.col,
            )


def _validate_measurement(measure: MeasureAST, qregs: Mapping[str, int], cregs: Mapping[str, int]) -> None:
    _validate_qargs([measure.q], qregs)
    cref = measure.c
    size = cregs.get(cref.reg)
    if size is None:
        raise QasmError(
            "E402",
            f"Classical register '{cref.reg}' is not defined.",
            cref.line,
            cref.col,
        )
    if cref.idx is not None and (cref.idx < 0 or cref.idx >= size):
        raise QasmError(
            "E402",
            f"Classical index {cref.idx} is out of range for register '{cref.reg}'.",
            cref.line,
            cref.col,
        )
    qsize = qregs.get(measure.q.reg)
    if measure.q.idx is None and cref.idx is None and qsize is not None and qsize != size:
        raise QasmError(
            "E402",
            f"Register sizes do not match for measurement '{measure.q.reg}' -> '{cref.reg}'.",
            measure.line,
            measure.col,
        )


def _assert_gate_known(call: GateCallAST, gate_defs: Mapping[str, GateDefAST]) -> None:
    if call.name in _BUILTIN_SIGNATURES:
        return
    if call.name in gate_defs:
        return
    raise QasmError(
        "E701",
        f"Gate '{call.name}' is not defined in the strict profile or user declarations.",
        call.line,
        call.col,
    )


def _assert_gate_signature(call: GateCallAST, gate_defs: Mapping[str, GateDefAST]) -> None:
    signature = _BUILTIN_SIGNATURES.get(call.name)
    if signature is None:
        definition = gate_defs.get(call.name)
        if definition is None:
            return
        signature = _GateSignature(len(definition.params), len(definition.qargs))
    if len(call.params) != signature.num_params:
        raise QasmError(
            "E703",
            f"Gate '{call.name}' expects {signature.num_params} parameter(s) but received {len(call.params)}.",
            call.line,
            call.col,
        )
    if len(call.qargs) != signature.num_qubits:
        raise QasmError(
            "E702",
            f"Gate '{call.name}' expects {signature.num_qubits} qubit operand(s) but received {len(call.qargs)}.",
            call.line,
            call.col,
        )
