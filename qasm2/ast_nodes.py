from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class QRef:
    """Reference to a quantum register or qubit.

    Parameters
    ----------
    reg : str
        Name of the quantum register.
    idx : Optional[int]
        Index within the register. ``None`` represents the full register.
    line : int
        Source line number where this reference appears.
    col : int
        Source column number where this reference appears.
    """

    reg: str
    idx: Optional[int]
    line: int
    col: int


@dataclass
class CRef:
    """Reference to a classical register or bit.

    Parameters
    ----------
    reg : str
        Name of the classical register.
    idx : Optional[int]
        Index within the register. ``None`` represents the full register.
    line : int
        Source line number where this reference appears.
    col : int
        Source column number where this reference appears.
    """

    reg: str
    idx: Optional[int]
    line: int
    col: int


@dataclass
class GateCallAST:
    """AST node representing a gate invocation.

    Parameters
    ----------
    name : str
        Name of the gate being invoked.
    params : List[float]
        Evaluated parameter values for the gate call.
    qargs : List[QRef]
        Quantum arguments passed to the gate.
    line : int
        Source line number of the gate call.
    col : int
        Source column number of the gate call.
    """

    name: str
    line: int
    col: int
    params: List[float] = field(default_factory=list)
    qargs: List[QRef] = field(default_factory=list)


@dataclass
class MeasureAST:
    """AST node representing a measurement operation.

    Parameters
    ----------
    q : QRef
        Quantum operand being measured.
    c : CRef
        Classical operand receiving the measurement result.
    line : int
        Source line number of the measurement.
    col : int
        Source column number of the measurement.
    """

    q: QRef
    c: CRef
    line: int
    col: int


@dataclass
class BarrierAST:
    """AST node representing a barrier.

    Parameters
    ----------
    qargs : List[QRef]
        Quantum arguments affected by the barrier.
    line : int
        Source line number of the barrier.
    col : int
        Source column number of the barrier.
    """

    line: int
    col: int
    qargs: List[QRef] = field(default_factory=list)


@dataclass
class GateDefAST:
    """AST node representing a user-defined gate declaration.

    Parameters
    ----------
    name : str
        Name of the user-defined gate.
    params : List[str]
        Parameter names for the gate definition.
    qargs : List[str]
        Quantum argument names for the gate definition.
    body : List[GateCallAST]
        Sequence of gate calls forming the body of the definition.
    line : int
        Source line number of the gate definition.
    col : int
        Source column number of the gate definition.
    """

    name: str
    line: int
    col: int
    params: List[str] = field(default_factory=list)
    qargs: List[str] = field(default_factory=list)
    body: List[GateCallAST] = field(default_factory=list)


@dataclass
class ProgramAST:
    """AST node representing a full OpenQASM 2 program.

    Parameters
    ----------
    version : str
        OpenQASM version string, typically ``"2.0"``.
    includes : List[str]
        Include directives encountered in the program.
    qregs : List[Tuple[str, int]]
        Declared quantum registers as name and size pairs.
    cregs : List[Tuple[str, int]]
        Declared classical registers as name and size pairs.
    gate_defs : Dict[str, GateDefAST]
        User-defined gate declarations indexed by gate name.
    body : List[Union[GateCallAST, MeasureAST, BarrierAST]]
        Sequence of top-level program statements.
    """

    version: str = "2.0"
    includes: List[str] = field(default_factory=list)
    qregs: List[Tuple[str, int]] = field(default_factory=list)
    cregs: List[Tuple[str, int]] = field(default_factory=list)
    gate_defs: Dict[str, GateDefAST] = field(default_factory=dict)
    body: List[Union[GateCallAST, MeasureAST, BarrierAST]] = field(default_factory=list)
