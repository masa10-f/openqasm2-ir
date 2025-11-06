"""Test suite for qasm2.normalize module.

This module tests the normalization utilities for OpenQASM 2 programs,
covering user-defined gate expansion, builtin gate alias normalization,
parameter substitution, quantum argument substitution, and barrier removal.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

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
from qasm2.normalize import (
    expand_gate_call,
    normalize_builtin_gate,
    normalize_program,
    substitute_params,
    substitute_qargs,
)


def _gate_def_call(
    name: str, params: list[Any], qargs: list[QRef], line: int = 1, col: int = 1
) -> GateCallAST:
    """Helper to create GateCallAST for use in gate definitions with symbolic params."""
    return GateCallAST(name=name, params=params, qargs=qargs, line=line, col=col)  # type: ignore[arg-type]


class TestUserDefinedGateExpansion:
    """Test expansion of user-defined gate macros."""

    def test_simple_single_gate_expansion(self) -> None:
        """Test expansion of a simple user-defined gate."""
        # Define gate: gate foo(a) q { rx(a) q; }
        foo_def = GateDefAST(
            name="foo",
            params=["a"],
            qargs=["q"],
            body=[
                _gate_def_call("rx", ["a"], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )
        gate_defs = {"foo": foo_def}

        # Call gate: foo(1.5) q[0];
        call = GateCallAST(
            name="foo",
            params=[1.5],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        expanded = expand_gate_call(call, gate_defs)

        assert len(expanded) == 1
        assert expanded[0].name == "rx"
        assert expanded[0].params == [1.5]
        assert expanded[0].qargs[0].reg == "q"
        assert expanded[0].qargs[0].idx == 0

    def test_nested_gate_expansion(self) -> None:
        """Test expansion of nested user-defined gates."""
        # Define gate: gate bar(b) q { foo(b) q; }
        # where foo is defined as above
        foo_def = GateDefAST(
            name="foo",
            params=["a"],
            qargs=["q"],
            body=[
                _gate_def_call("rx", ["a"], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )

        bar_def = GateDefAST(
            name="bar",
            params=["b"],
            qargs=["q"],
            body=[
                _gate_def_call("foo", ["b"], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )

        gate_defs = {"foo": foo_def, "bar": bar_def}

        # Call gate: bar(2.0) q[0];
        call = GateCallAST(
            name="bar",
            params=[2.0],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        expanded = expand_gate_call(call, gate_defs)

        # Should expand bar -> foo -> rx
        assert len(expanded) == 1
        assert expanded[0].name == "rx"
        assert expanded[0].params == [2.0]

    def test_multi_qubit_gate_expansion(self) -> None:
        """Test expansion of multi-qubit user-defined gates."""
        # Define gate: gate controlled_x(q1, q2) { cx q1, q2; }
        gate_def = GateDefAST(
            name="controlled_x",
            params=[],
            qargs=["q1", "q2"],
            body=[
                GateCallAST(
                    name="cx", params=[], qargs=[QRef("q1", None, 1, 1), QRef("q2", None, 1, 1)], line=1, col=1
                ),
            ],
            line=1,
            col=1,
        )

        gate_defs = {"controlled_x": gate_def}

        # Call gate: controlled_x q[0], q[1];
        call = GateCallAST(
            name="controlled_x",
            params=[],
            qargs=[QRef("q", 0, 1, 1), QRef("q", 1, 1, 1)],
            line=1,
            col=1,
        )

        expanded = expand_gate_call(call, gate_defs)

        assert len(expanded) == 1
        assert expanded[0].name == "cx"
        assert len(expanded[0].qargs) == 2
        assert expanded[0].qargs[0].reg == "q"
        assert expanded[0].qargs[0].idx == 0
        assert expanded[0].qargs[1].reg == "q"
        assert expanded[0].qargs[1].idx == 1

    def test_builtin_gate_no_expansion(self) -> None:
        """Test that builtin gates are not expanded."""
        gate_defs: Dict[str, GateDefAST] = {}

        call = GateCallAST(
            name="rx",
            params=[1.5],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        expanded = expand_gate_call(call, gate_defs)

        assert len(expanded) == 1
        assert expanded[0].name == "rx"
        assert expanded[0] is not call  # Should be a copy

    def test_expansion_parameter_count_mismatch(self) -> None:
        """Test error when gate parameters don't match."""
        gate_def = GateDefAST(
            name="foo",
            params=["a", "b"],
            qargs=["q"],
            body=[],
            line=1,
            col=1,
        )

        gate_defs = {"foo": gate_def}

        # Call with wrong number of parameters
        call = GateCallAST(
            name="foo",
            params=[1.5],  # Only one parameter, but gate expects two
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            expand_gate_call(call, gate_defs)
        assert "E702" in str(excinfo.value)

    def test_expansion_qubit_count_mismatch(self) -> None:
        """Test error when gate qubit arguments don't match."""
        gate_def = GateDefAST(
            name="foo",
            params=[],
            qargs=["q1", "q2"],
            body=[],
            line=1,
            col=1,
        )

        gate_defs = {"foo": gate_def}

        # Call with wrong number of qubits
        call = GateCallAST(
            name="foo",
            params=[],
            qargs=[QRef("q", 0, 1, 1)],  # Only one qubit, but gate expects two
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            expand_gate_call(call, gate_defs)
        assert "E703" in str(excinfo.value)

    def test_expansion_recursion_limit(self) -> None:
        """Test that excessive recursion depth is caught."""
        # Create a recursive gate definition that would cause infinite expansion
        gate_def = GateDefAST(
            name="foo",
            params=[],
            qargs=["q"],
            body=[
                _gate_def_call("foo", [], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )

        gate_defs = {"foo": gate_def}

        call = GateCallAST(
            name="foo",
            params=[],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            expand_gate_call(call, gate_defs)
        assert "E701" in str(excinfo.value)
        assert "maximum depth" in str(excinfo.value)


class TestBuiltinGateNormalization:
    """Test normalization of builtin gate aliases."""

    def test_u1_to_rz(self) -> None:
        """Test normalization of u1 gate to rz."""
        call = GateCallAST(
            name="u1",
            params=[1.5],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        assert len(normalized) == 1
        assert normalized[0].name == "rz"
        assert normalized[0].params == [1.5]

    def test_u2_decomposition(self) -> None:
        """Test decomposition of u2 gate."""
        phi = 1.0
        lam = 2.0
        call = GateCallAST(
            name="u2",
            params=[phi, lam],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        # u2 should decompose into rz-rx-rz
        assert len(normalized) == 3
        assert normalized[0].name == "rz"
        assert normalized[1].name == "rx"
        assert normalized[2].name == "rz"

        # Check parameter calculations
        half_pi = math.pi / 2.0
        assert normalized[0].params[0] == pytest.approx(phi + half_pi)
        assert normalized[1].params[0] == pytest.approx(half_pi)
        assert normalized[2].params[0] == pytest.approx(lam - half_pi)

    def test_u3_decomposition(self) -> None:
        """Test decomposition of u3 gate."""
        theta = 0.5
        phi = 1.0
        lam = 1.5
        call = GateCallAST(
            name="u3",
            params=[theta, phi, lam],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        # u3 should decompose into rz-rx-rz
        assert len(normalized) == 3
        assert normalized[0].name == "rz"
        assert normalized[1].name == "rx"
        assert normalized[2].name == "rz"

        # Check parameter assignments
        assert normalized[0].params[0] == lam
        assert normalized[1].params[0] == theta
        assert normalized[2].params[0] == phi

    def test_p_to_rz(self) -> None:
        """Test normalization of p gate to rz."""
        call = GateCallAST(
            name="p",
            params=[2.5],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        assert len(normalized) == 1
        assert normalized[0].name == "rz"
        assert normalized[0].params == [2.5]

    def test_cu1_to_crz(self) -> None:
        """Test normalization of cu1 gate to crz."""
        call = GateCallAST(
            name="cu1",
            params=[1.5],
            qargs=[QRef("q", 0, 1, 1), QRef("q", 1, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        assert len(normalized) == 1
        assert normalized[0].name == "crz"
        assert normalized[0].params == [1.5]

    def test_primitive_gate_unchanged(self) -> None:
        """Test that primitive gates remain unchanged."""
        call = GateCallAST(
            name="x",
            params=[],
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        normalized = normalize_builtin_gate(call)

        assert len(normalized) == 1
        assert normalized[0].name == "x"
        assert normalized[0] is not call  # Should be a copy

    def test_u1_wrong_parameter_count(self) -> None:
        """Test error when u1 has wrong parameter count."""
        call = GateCallAST(
            name="u1",
            params=[1.0, 2.0],  # u1 expects 1 parameter
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            normalize_builtin_gate(call)
        assert "E702" in str(excinfo.value)

    def test_u2_wrong_parameter_count(self) -> None:
        """Test error when u2 has wrong parameter count."""
        call = GateCallAST(
            name="u2",
            params=[1.0],  # u2 expects 2 parameters
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            normalize_builtin_gate(call)
        assert "E702" in str(excinfo.value)

    def test_u3_wrong_parameter_count(self) -> None:
        """Test error when u3 has wrong parameter count."""
        call = GateCallAST(
            name="u3",
            params=[1.0, 2.0],  # u3 expects 3 parameters
            qargs=[QRef("q", 0, 1, 1)],
            line=1,
            col=1,
        )

        with pytest.raises(QasmError) as excinfo:
            normalize_builtin_gate(call)
        assert "E702" in str(excinfo.value)


class TestParameterSubstitution:
    """Test parameter substitution in gate bodies."""

    def test_numeric_parameter_passthrough(self) -> None:
        """Test that numeric parameters pass through unchanged."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", None, 1, 1)]),
        ]

        param_map: Dict[str, float] = {}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 1
        assert substituted[0].params == [1.5]

    def test_symbolic_parameter_substitution(self) -> None:
        """Test substitution of symbolic parameters."""
        body = [
            _gate_def_call("rx", ["a"], [QRef("q", None, 1, 1)]),
        ]

        param_map = {"a": 2.5}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 1
        assert substituted[0].params == [2.5]

    def test_multiple_parameter_substitution(self) -> None:
        """Test substitution of multiple parameters."""
        body = [
            _gate_def_call("u3", ["a", "b", "c"], [QRef("q", None, 1, 1)]),
        ]

        param_map = {"a": 1.0, "b": 2.0, "c": 3.0}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 1
        assert substituted[0].params == [1.0, 2.0, 3.0]

    def test_mixed_numeric_and_symbolic(self) -> None:
        """Test substitution with mix of numeric and symbolic parameters."""
        body = [
            _gate_def_call("u3", [1.5, "b", 3.0], [QRef("q", None, 1, 1)]),
        ]

        param_map = {"b": 2.5}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 1
        assert substituted[0].params == [1.5, 2.5, 3.0]

    def test_undefined_parameter_error(self) -> None:
        """Test error when symbolic parameter is not defined."""
        body = [
            _gate_def_call("rx", ["undefined"], [QRef("q", None, 1, 1)]),
        ]

        param_map: Dict[str, float] = {}

        with pytest.raises(QasmError) as excinfo:
            substitute_params(body, param_map)
        assert "E702" in str(excinfo.value)
        assert "not defined" in str(excinfo.value)

    def test_integer_parameter_conversion(self) -> None:
        """Test that integer parameters are converted to float."""
        body = [
            _gate_def_call("rx", [42], [QRef("q", None, 1, 1)]),
        ]

        param_map: Dict[str, float] = {}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 1
        assert substituted[0].params == [42.0]
        assert isinstance(substituted[0].params[0], float)

    def test_multiple_gate_parameter_substitution(self) -> None:
        """Test substitution across multiple gates."""
        body = [
            _gate_def_call("rx", ["a"], [QRef("q", None, 1, 1)]),
            _gate_def_call("rz", ["b"], [QRef("q", None, 1, 1)]),
            _gate_def_call("ry", ["a"], [QRef("q", None, 1, 1)]),
        ]

        param_map = {"a": 1.0, "b": 2.0}
        substituted = substitute_params(body, param_map)

        assert len(substituted) == 3
        assert substituted[0].params == [1.0]
        assert substituted[1].params == [2.0]
        assert substituted[2].params == [1.0]


class TestQuantumArgumentSubstitution:
    """Test quantum argument substitution in gate bodies."""

    def test_single_qubit_substitution(self) -> None:
        """Test substitution of single-qubit arguments."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", None, 1, 1)]),
        ]

        qarg_map = {"q": QRef("a", 0, 1, 1)}
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 1
        assert substituted[0].qargs[0].reg == "a"
        assert substituted[0].qargs[0].idx == 0

    def test_multi_qubit_substitution(self) -> None:
        """Test substitution of multi-qubit arguments."""
        body = [
            _gate_def_call("cx", [], [QRef("q1", None, 1, 1), QRef("q2", None, 1, 1)]),
        ]

        qarg_map = {
            "q1": QRef("q", 0, 1, 1),
            "q2": QRef("q", 1, 1, 1),
        }
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 1
        assert len(substituted[0].qargs) == 2
        assert substituted[0].qargs[0].reg == "q"
        assert substituted[0].qargs[0].idx == 0
        assert substituted[0].qargs[1].reg == "q"
        assert substituted[0].qargs[1].idx == 1

    def test_indexed_to_indexed_substitution(self) -> None:
        """Test substitution of indexed qubit references."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", 0, 1, 1)]),
        ]

        qarg_map = {"q": QRef("a", 2, 1, 1)}
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 1
        assert substituted[0].qargs[0].reg == "a"
        assert substituted[0].qargs[0].idx == 2

    def test_register_to_register_substitution(self) -> None:
        """Test substitution of whole register references."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", None, 1, 1)]),
        ]

        qarg_map = {"q": QRef("a", None, 1, 1)}
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 1
        assert substituted[0].qargs[0].reg == "a"
        assert substituted[0].qargs[0].idx is None

    def test_undefined_qarg_error(self) -> None:
        """Test error when quantum argument is not defined."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("undefined", None, 1, 1)]),
        ]

        qarg_map: Dict[str, QRef] = {}

        with pytest.raises(QasmError) as excinfo:
            substitute_qargs(body, qarg_map)
        assert "E703" in str(excinfo.value)
        assert "not defined" in str(excinfo.value)

    def test_multiple_gate_qarg_substitution(self) -> None:
        """Test substitution across multiple gates."""
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", None, 1, 1)]),
            _gate_def_call("rz", [2.0], [QRef("q", None, 1, 1)]),
        ]

        qarg_map = {"q": QRef("a", 0, 1, 1)}
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 2
        assert all(qarg.reg == "a" for gate in substituted for qarg in gate.qargs)

    def test_indexed_zero_special_case(self) -> None:
        """Test that index 0 is treated specially in template matching."""
        # When template has idx=0 and actual has idx=None, use template's idx (0)
        body = [
            _gate_def_call("rx", [1.5], [QRef("q", 0, 1, 1)]),
        ]

        qarg_map = {"q": QRef("a", None, 1, 1)}
        substituted = substitute_qargs(body, qarg_map)

        assert len(substituted) == 1
        assert substituted[0].qargs[0].reg == "a"
        assert substituted[0].qargs[0].idx == 0


class TestBarrierRemoval:
    """Test removal of barrier statements during normalization."""

    def test_barrier_removal_simple(self) -> None:
        """Test that barriers are removed during normalization."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
                GateCallAST(name="y", params=[], qargs=[QRef("q", 0, 1, 1)], line=3, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 2
        assert isinstance(normalized.body[0], GateCallAST)
        assert isinstance(normalized.body[1], GateCallAST)
        assert normalized.body[0].name == "x"
        assert normalized.body[1].name == "y"

    def test_multiple_barriers_removal(self) -> None:
        """Test removal of multiple barrier statements."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            body=[
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
                GateCallAST(name="y", params=[], qargs=[QRef("q", 1, 1, 1)], line=3, col=1),
                BarrierAST(qargs=[QRef("q", 1, 1, 1)], line=4, col=1),
                GateCallAST(name="z", params=[], qargs=[QRef("q", 0, 1, 1)], line=5, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 3
        assert all(isinstance(stmt, GateCallAST) for stmt in normalized.body)

    def test_barrier_only_program(self) -> None:
        """Test normalization of program with only barriers."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 0


class TestProgramNormalization:
    """Test complete program normalization."""

    def test_simple_program_normalization(self) -> None:
        """Test normalization of a simple program."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                GateCallAST(name="y", params=[], qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 2
        assert all(isinstance(stmt, GateCallAST) for stmt in normalized.body)
        assert normalized.gate_defs == {}

    def test_program_with_user_gate_and_barrier(self) -> None:
        """Test normalization with user-defined gates and barriers."""
        # Define gate: gate foo(a) q { rx(a) q; }
        foo_def = GateDefAST(
            name="foo",
            params=["a"],
            qargs=["q"],
            body=[
                _gate_def_call("rx", ["a"], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )

        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            gate_defs={"foo": foo_def},
            body=[
                GateCallAST(name="foo", params=[1.5], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=3, col=1),
            ],
        )

        normalized = normalize_program(program)

        # Should have 2 statements: expanded foo (which is rx), and x (barrier removed)
        assert len(normalized.body) == 2
        assert isinstance(normalized.body[0], GateCallAST)
        assert isinstance(normalized.body[1], GateCallAST)
        assert normalized.body[0].name == "rx"
        assert normalized.body[1].name == "x"
        assert normalized.gate_defs == {}

    def test_program_with_measurement(self) -> None:
        """Test normalization of program with measurements."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            cregs=[("c", 1)],
            body=[
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                MeasureAST(q=QRef("q", 0, 1, 1), c=CRef("c", 0, 1, 1), line=2, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 2
        assert isinstance(normalized.body[0], GateCallAST)
        assert isinstance(normalized.body[1], MeasureAST)

    def test_program_with_u1_normalization(self) -> None:
        """Test normalization of program with u1 gates."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(name="u1", params=[1.5], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 1
        assert isinstance(normalized.body[0], GateCallAST)
        assert normalized.body[0].name == "rz"
        assert normalized.body[0].params == [1.5]

    def test_program_with_u3_decomposition(self) -> None:
        """Test normalization of program with u3 gates."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(name="u3", params=[0.5, 1.0, 1.5], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
            ],
        )

        normalized = normalize_program(program)

        # u3 should expand to rz-rx-rz
        assert len(normalized.body) == 3
        assert isinstance(normalized.body[0], GateCallAST)
        assert isinstance(normalized.body[1], GateCallAST)
        assert isinstance(normalized.body[2], GateCallAST)
        assert normalized.body[0].name == "rz"
        assert normalized.body[1].name == "rx"
        assert normalized.body[2].name == "rz"

    def test_program_metadata_preservation(self) -> None:
        """Test that program metadata is preserved during normalization."""
        program = ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 2), ("r", 1)],
            cregs=[("c", 2)],
            body=[
                GateCallAST(name="x", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
            ],
        )

        normalized = normalize_program(program)

        assert normalized.version == "2.0"
        assert normalized.includes == ["qelib1.inc"]
        assert normalized.qregs == [("q", 2), ("r", 1)]
        assert normalized.cregs == [("c", 2)]

    def test_empty_program(self) -> None:
        """Test normalization of empty program."""
        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[],
        )

        normalized = normalize_program(program)

        assert len(normalized.body) == 0
        assert normalized.gate_defs == {}

    def test_complex_program_combination(self) -> None:
        """Test normalization of complex program with multiple features."""
        # Define a parametric gate
        param_gate = GateDefAST(
            name="rxy",
            params=["angle"],
            qargs=["q"],
            body=[
                _gate_def_call("rx", ["angle"], [QRef("q", None, 1, 1)]),
                _gate_def_call("ry", ["angle"], [QRef("q", None, 1, 1)]),
            ],
            line=1,
            col=1,
        )

        program = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            cregs=[("c", 1)],
            gate_defs={"rxy": param_gate},
            body=[
                GateCallAST(name="h", params=[], qargs=[QRef("q", 0, 1, 1)], line=1, col=1),
                BarrierAST(qargs=[QRef("q", 0, 1, 1)], line=2, col=1),
                GateCallAST(name="rxy", params=[math.pi / 2], qargs=[QRef("q", 0, 1, 1)], line=3, col=1),
                GateCallAST(name="u1", params=[1.5], qargs=[QRef("q", 0, 1, 1)], line=4, col=1),
                MeasureAST(q=QRef("q", 0, 1, 1), c=CRef("c", 0, 1, 1), line=5, col=1),
            ],
        )

        normalized = normalize_program(program)

        # Expectation: h, rxy expands to rx+ry, u1 normalizes to rz, measure
        # That's 1 + 2 + 1 + 1 = 5 statements
        assert len(normalized.body) == 5
        assert isinstance(normalized.body[0], GateCallAST)
        assert isinstance(normalized.body[1], GateCallAST)
        assert isinstance(normalized.body[2], GateCallAST)
        assert isinstance(normalized.body[3], GateCallAST)
        assert normalized.body[0].name == "h"
        assert normalized.body[1].name == "rx"
        assert normalized.body[2].name == "ry"
        assert normalized.body[3].name == "rz"
        assert isinstance(normalized.body[4], MeasureAST)
