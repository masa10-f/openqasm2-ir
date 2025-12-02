"""Normalization utilities for the validated OpenQASM 2 AST.

This module expands user-defined gate macros, normalises builtin gate aliases,
and removes syntactic artefacts such as barriers. The resulting program AST is
guaranteed to contain only primitive gate calls and measurement operations in
its body, with user-defined gate declarations erased.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Union

from qasm2.ast_nodes import (
    BarrierAST,
    CRef,
    GateCallAST,
    GateDefAST,
    MeasureAST,
    ProgramAST,
    QRef,
)
from qasm2.errors import QasmError

__all__ = [
    "normalize_program",
    "expand_gate_call",
    "normalize_builtin_gate",
    "substitute_params",
    "substitute_qargs",
]

_MAX_EXPANSION_DEPTH = 100


def normalize_program(ast: ProgramAST) -> ProgramAST:
    """Normalize a validated program AST.

    Parameters
    ----------
    ast
        Validated OpenQASM 2 program.

    Returns
    -------
    ProgramAST
        Normalized AST with expanded gate macros and canonical gate calls.
    """
    gate_defs = dict(ast.gate_defs)
    normalized_body: List[Union[GateCallAST, MeasureAST, BarrierAST]] = []

    for stmt in ast.body:
        if isinstance(stmt, BarrierAST):
            continue
        if isinstance(stmt, MeasureAST):
            normalized_body.append(_clone_measure(stmt))
            continue
        if isinstance(stmt, GateCallAST):
            expanded_calls = expand_gate_call(stmt, gate_defs)
            for call in expanded_calls:
                normalized_calls = normalize_builtin_gate(call)
                normalized_body.extend(normalized_calls)
            continue
        raise QasmError(
            "E701",
            "Unsupported statement encountered during normalization.",
            getattr(stmt, "line", 1),
            getattr(stmt, "col", 1),
        )

    return ProgramAST(
        version=ast.version,
        includes=list(ast.includes),
        qregs=list(ast.qregs),
        cregs=list(ast.cregs),
        gate_defs={},
        body=normalized_body,
    )


def expand_gate_call(call: GateCallAST, gate_defs: Dict[str, GateDefAST], depth: int = 0) -> List[GateCallAST]:
    """Inline user-defined gate invocations.

    Parameters
    ----------
    call
        Gate call to expand.
    gate_defs
        Mapping of user-defined gate declarations.
    depth
        Current expansion depth used to guard against runaway recursion.

    Returns
    -------
    List[GateCallAST]
        Sequence of gate calls after expansion. Built-in gates are returned as-is.

    Raises
    ------
    QasmError
        If the expansion exceeds the permitted recursion depth or encounters
        inconsistent gate signatures.
    """
    if depth > _MAX_EXPANSION_DEPTH:
        raise QasmError(
            "E701",
            f"Gate expansion exceeded maximum depth of {_MAX_EXPANSION_DEPTH}.",
            call.line,
            call.col,
        )

    definition = gate_defs.get(call.name)
    if definition is None:
        return [_clone_gate_call(call)]

    if len(call.params) != len(definition.params):
        raise QasmError(
            "E702",
            f"Gate '{call.name}' expects {len(definition.params)} parameters but received {len(call.params)}.",
            call.line,
            call.col,
        )
    if len(call.qargs) != len(definition.qargs):
        raise QasmError(
            "E703",
            f"Gate '{call.name}' expects {len(definition.qargs)} qubit operands but received {len(call.qargs)}.",
            call.line,
            call.col,
        )

    param_map = {param_name: float(value) for param_name, value in zip(definition.params, call.params)}
    qarg_map = {qarg_name: actual for qarg_name, actual in zip(definition.qargs, call.qargs)}

    substituted = substitute_params(definition.body, param_map)
    substituted = substitute_qargs(substituted, qarg_map)

    expanded_calls: List[GateCallAST] = []
    for nested_call in substituted:
        expanded_calls.extend(expand_gate_call(nested_call, gate_defs, depth + 1))
    return expanded_calls


def normalize_builtin_gate(call: GateCallAST) -> List[GateCallAST]:
    """Normalize builtin gate aliases into canonical gate sequences."""
    name = call.name
    if name == "u1":
        _assert_arity(call, 1)
        return [_clone_gate_call(call, name="rz")]
    if name == "u2":
        _assert_arity(call, 2)
        phi, lam = call.params
        half_pi = math.pi / 2.0
        return [
            _clone_gate_call(call, name="rz", params=[phi + half_pi]),
            _clone_gate_call(call, name="rx", params=[half_pi]),
            _clone_gate_call(call, name="rz", params=[lam - half_pi]),
        ]
    if name == "u3":
        # u3 is natively supported in GraphQOMB, pass through without decomposition
        _assert_arity(call, 3)
        return [_clone_gate_call(call)]
    if name == "p":
        _assert_arity(call, 1)
        return [_clone_gate_call(call, name="rz")]
    if name == "cu1":
        _assert_arity(call, 1)
        return [_clone_gate_call(call, name="crz")]
    return [_clone_gate_call(call)]


def substitute_params(body: Sequence[GateCallAST], param_map: Dict[str, float]) -> List[GateCallAST]:
    """Replace parameter placeholders within a gate body."""
    substituted: List[GateCallAST] = []
    for call in body:
        resolved_params: List[float] = []
        for raw_value in call.params:
            if isinstance(raw_value, (int, float)):
                resolved_params.append(float(raw_value))
                continue
            if isinstance(raw_value, str):
                if raw_value not in param_map:
                    raise QasmError(
                        "E702",
                        f"Parameter '{raw_value}' is not defined for gate.",
                        call.line,
                        call.col,
                    )
                resolved_params.append(float(param_map[raw_value]))
                continue
            raise QasmError(
                "E702",
                "Unsupported parameter type encountered during substitution.",
                call.line,
                call.col,
            )

        substituted.append(
            GateCallAST(
                name=call.name,
                line=call.line,
                col=call.col,
                params=resolved_params,
                qargs=[_clone_qref(qarg) for qarg in call.qargs],
            )
        )
    return substituted


def substitute_qargs(body: Sequence[GateCallAST], qarg_map: Dict[str, QRef]) -> List[GateCallAST]:
    """Replace symbolic qubit references with concrete operands."""
    substituted: List[GateCallAST] = []
    for call in body:
        resolved_qargs: List[QRef] = []
        for qref in call.qargs:
            mapped = qarg_map.get(qref.reg)
            if mapped is None:
                raise QasmError(
                    "E703",
                    f"Qubit argument '{qref.reg}' is not defined for gate.",
                    qref.line,
                    qref.col,
                )
            resolved_qargs.append(_resolve_qref(qref, mapped))

        substituted.append(
            GateCallAST(
                name=call.name,
                line=call.line,
                col=call.col,
                params=list(call.params),
                qargs=resolved_qargs,
            )
        )
    return substituted


def _clone_gate_call(
    call: GateCallAST,
    name: str | None = None,
    params: Iterable[float] | None = None,
    qargs: Iterable[QRef] | None = None,
) -> GateCallAST:
    """Create a defensive copy of a gate call."""
    copied_qargs = [_clone_qref(qref) for qref in (qargs or call.qargs)]
    copied_params = [float(value) for value in (params or call.params)]
    return GateCallAST(
        name=name or call.name,
        line=call.line,
        col=call.col,
        params=copied_params,
        qargs=copied_qargs,
    )


def _clone_measure(node: MeasureAST) -> MeasureAST:
    """Create a defensive copy of a measurement statement."""
    return MeasureAST(
        q=_clone_qref(node.q),
        c=_clone_cref(node.c),
        line=node.line,
        col=node.col,
    )


def _clone_qref(ref: QRef) -> QRef:
    """Clone a quantum reference node."""
    return QRef(reg=ref.reg, idx=ref.idx, line=ref.line, col=ref.col)


def _clone_cref(ref: CRef) -> CRef:
    """Clone a classical reference node."""
    return CRef(reg=ref.reg, idx=ref.idx, line=ref.line, col=ref.col)


def _resolve_qref(template: QRef, actual: QRef) -> QRef:
    """Resolve a gate-local quantum reference to a concrete operand."""
    if template.idx is None:
        return QRef(reg=actual.reg, idx=actual.idx, line=template.line, col=template.col)

    if actual.idx is None:
        return QRef(reg=actual.reg, idx=template.idx, line=template.line, col=template.col)

    if template.idx not in (0, actual.idx):
        raise QasmError(
            "E703",
            "Invalid indexed access on a single-qubit argument.",
            template.line,
            template.col,
        )
    return QRef(reg=actual.reg, idx=actual.idx, line=template.line, col=template.col)


def _assert_arity(call: GateCallAST, expected: int) -> None:
    """Ensure the gate call contains the expected number of parameters."""
    if len(call.params) != expected:
        raise QasmError(
            "E702",
            f"Gate '{call.name}' expects {expected} parameters but received {len(call.params)}.",
            call.line,
            call.col,
        )
