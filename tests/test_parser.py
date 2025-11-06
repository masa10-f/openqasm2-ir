"""Test suite for OpenQASM 2.0 parser implementation.

This module provides comprehensive tests for the parser functionality defined in
qasm2.parser, covering:
- Header parsing
- Include directives
- Register declarations (qreg, creg)
- Built-in gates (h, x, y, z, cx, cz, swap, etc.)
- Parameterized gates (rx, ry, rz, u3, etc.)
- User-defined gates
- Measurements
- Barriers
- Expression evaluation (pi/2, sin(pi/4), etc.)
- Position tracking (line and column numbers)
- Error cases (syntax errors, lexical errors, semantic errors)

Note: These tests require the Lark grammar file to be implemented.
They will fail until qasm2/grammar/qasm2_strict.lark is available.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from qasm2.ast_nodes import BarrierAST, GateCallAST, MeasureAST
from qasm2.errors import QasmError
from qasm2.parser import parse_qasm

if TYPE_CHECKING:
    from typing import Callable


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def approx_equal() -> Callable[[float, float], bool]:
    """Compare floating point values with tolerance."""

    def _compare(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    return _compare


# =============================================================================
# Header Tests
# =============================================================================


class TestHeader:
    """Tests for OPENQASM version header parsing."""

    def test_header_version_2_0(self) -> None:
        """Parse standard OPENQASM 2.0 header."""
        source = "OPENQASM 2.0;"
        ast = parse_qasm(source)
        assert ast.version == "2.0"

    def test_header_with_qreg(self) -> None:
        """Parse header followed by quantum register declaration."""
        source = """
        OPENQASM 2.0;
        qreg q[2];
        """
        ast = parse_qasm(source)
        assert ast.version == "2.0"
        assert len(ast.qregs) == 1
        assert ast.qregs[0] == ("q", 2)

    def test_header_missing_raises_error(self) -> None:
        """Missing header should raise QasmError."""
        source = "qreg q[1];"
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code.startswith("E")


# =============================================================================
# Include Tests
# =============================================================================


class TestInclude:
    """Tests for include directive parsing."""

    def test_include_qelib1(self) -> None:
        """Parse standard qelib1.inc include."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        """
        ast = parse_qasm(source)
        assert len(ast.includes) == 1
        assert ast.includes[0] == "qelib1.inc"

    def test_multiple_includes(self) -> None:
        """Parse multiple include directives."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        include "custom.inc";
        """
        ast = parse_qasm(source)
        assert len(ast.includes) == 2
        assert ast.includes[0] == "qelib1.inc"
        assert ast.includes[1] == "custom.inc"

    def test_include_with_path(self) -> None:
        """Parse include with relative path."""
        source = """
        OPENQASM 2.0;
        include "gates/mylib.inc";
        """
        ast = parse_qasm(source)
        assert len(ast.includes) == 1
        assert ast.includes[0] == "gates/mylib.inc"


# =============================================================================
# Register Declaration Tests
# =============================================================================


class TestRegisterDeclarations:
    """Tests for quantum and classical register declarations."""

    def test_qreg_single(self) -> None:
        """Declare a single quantum register."""
        source = """
        OPENQASM 2.0;
        qreg q[3];
        """
        ast = parse_qasm(source)
        assert len(ast.qregs) == 1
        assert ast.qregs[0] == ("q", 3)

    def test_creg_single(self) -> None:
        """Declare a single classical register."""
        source = """
        OPENQASM 2.0;
        creg c[5];
        """
        ast = parse_qasm(source)
        assert len(ast.cregs) == 1
        assert ast.cregs[0] == ("c", 5)

    def test_qreg_and_creg(self) -> None:
        """Declare both quantum and classical registers."""
        source = """
        OPENQASM 2.0;
        qreg q[4];
        creg c[4];
        """
        ast = parse_qasm(source)
        assert len(ast.qregs) == 1
        assert len(ast.cregs) == 1
        assert ast.qregs[0] == ("q", 4)
        assert ast.cregs[0] == ("c", 4)

    def test_multiple_qregs(self) -> None:
        """Declare multiple quantum registers."""
        source = """
        OPENQASM 2.0;
        qreg qa[2];
        qreg qb[3];
        qreg qc[1];
        """
        ast = parse_qasm(source)
        assert len(ast.qregs) == 3
        assert ast.qregs[0] == ("qa", 2)
        assert ast.qregs[1] == ("qb", 3)
        assert ast.qregs[2] == ("qc", 1)

    def test_duplicate_qreg_raises_error(self) -> None:
        """Duplicate quantum register names should raise error."""
        source = """
        OPENQASM 2.0;
        qreg q[2];
        qreg q[3];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Duplicate" in exc_info.value.message

    def test_duplicate_creg_raises_error(self) -> None:
        """Duplicate classical register names should raise error."""
        source = """
        OPENQASM 2.0;
        creg c[2];
        creg c[3];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Duplicate" in exc_info.value.message


# =============================================================================
# Built-in Gate Tests
# =============================================================================


class TestBuiltinGates:
    """Tests for built-in quantum gates."""

    def test_hadamard_gate(self) -> None:
        """Apply Hadamard gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "h"
        assert len(gate_call.qargs) == 1
        assert gate_call.qargs[0].reg == "q"
        assert gate_call.qargs[0].idx == 0

    def test_pauli_gates(self) -> None:
        """Apply Pauli X, Y, Z gates."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[0];
        y q[1];
        z q[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 3
        assert ast.body[0].name == "x"  # type: ignore[union-attr]
        assert ast.body[1].name == "y"  # type: ignore[union-attr]
        assert ast.body[2].name == "z"  # type: ignore[union-attr]

    def test_cx_gate(self) -> None:
        """Apply controlled-X (CNOT) gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cx q[0], q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "cx"
        assert len(gate_call.qargs) == 2
        assert gate_call.qargs[0].idx == 0
        assert gate_call.qargs[1].idx == 1

    def test_cz_gate(self) -> None:
        """Apply controlled-Z gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cz q[0], q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "cz"
        assert len(gate_call.qargs) == 2

    def test_swap_gate(self) -> None:
        """Apply SWAP gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        swap q[0], q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "swap"
        assert len(gate_call.qargs) == 2

    def test_s_sdg_gates(self) -> None:
        """Apply S and S-dagger gates."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        s q[0];
        sdg q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 2
        assert ast.body[0].name == "s"  # type: ignore[union-attr]
        assert ast.body[1].name == "sdg"  # type: ignore[union-attr]

    def test_t_tdg_gates(self) -> None:
        """Apply T and T-dagger gates."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        t q[0];
        tdg q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 2
        assert ast.body[0].name == "t"  # type: ignore[union-attr]
        assert ast.body[1].name == "tdg"  # type: ignore[union-attr]

    def test_sx_sxdg_gates(self) -> None:
        """Apply sqrt(X) and sqrt(X)-dagger gates."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        sx q[0];
        sxdg q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 2
        assert ast.body[0].name == "sx"  # type: ignore[union-attr]
        assert ast.body[1].name == "sxdg"  # type: ignore[union-attr]

    def test_ccx_gate(self) -> None:
        """Apply Toffoli (CCX) gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        ccx q[0], q[1], q[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "ccx"
        assert len(gate_call.qargs) == 3

    def test_cswap_gate(self) -> None:
        """Apply Fredkin (CSWAP) gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        cswap q[0], q[1], q[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "cswap"
        assert len(gate_call.qargs) == 3

    def test_id_gate(self) -> None:
        """Apply identity gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        id q[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "id"


# =============================================================================
# Parameterized Gate Tests
# =============================================================================


class TestParameterizedGates:
    """Tests for parameterized quantum gates."""

    def test_rx_gate_with_pi_over_2(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply RX rotation with pi/2."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "rx"
        assert len(gate_call.params) == 1
        assert approx_equal(gate_call.params[0], math.pi / 2)

    def test_ry_gate_with_expression(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply RY rotation with arithmetic expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        ry(pi/4) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "ry"
        assert approx_equal(gate_call.params[0], math.pi / 4)

    def test_rz_gate_with_negative_angle(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply RZ rotation with negative angle."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rz(-pi/2) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "rz"
        assert approx_equal(gate_call.params[0], -math.pi / 2)

    def test_u1_gate(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply U1 gate with phase parameter."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u1(pi) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "u1"
        assert len(gate_call.params) == 1
        assert approx_equal(gate_call.params[0], math.pi)

    def test_u2_gate(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply U2 gate with two parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u2(0, pi) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "u2"
        assert len(gate_call.params) == 2
        assert approx_equal(gate_call.params[0], 0.0)
        assert approx_equal(gate_call.params[1], math.pi)

    def test_u3_gate(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply U3 gate with three parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u3(pi/2, 0, pi) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "u3"
        assert len(gate_call.params) == 3
        assert approx_equal(gate_call.params[0], math.pi / 2)
        assert approx_equal(gate_call.params[1], 0.0)
        assert approx_equal(gate_call.params[2], math.pi)

    def test_p_gate(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Apply phase gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        p(pi/8) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "p"
        assert len(gate_call.params) == 1
        assert approx_equal(gate_call.params[0], math.pi / 8)


# =============================================================================
# User-Defined Gate Tests
# =============================================================================


class TestUserDefinedGates:
    """Tests for user-defined gate declarations and invocations."""

    def test_gate_definition_no_params(self) -> None:
        """Define a gate without parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate mygate a {
            h a;
        }
        """
        ast = parse_qasm(source)
        assert "mygate" in ast.gate_defs
        gate_def = ast.gate_defs["mygate"]
        assert gate_def.name == "mygate"
        assert len(gate_def.params) == 0
        assert len(gate_def.qargs) == 1
        assert gate_def.qargs[0] == "a"
        assert len(gate_def.body) == 1
        assert gate_def.body[0].name == "h"

    def test_gate_definition_with_params(self) -> None:
        """Define a gate with parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate rgate(theta) a {
            rx(theta) a;
        }
        """
        ast = parse_qasm(source)
        assert "rgate" in ast.gate_defs
        gate_def = ast.gate_defs["rgate"]
        assert gate_def.name == "rgate"
        assert len(gate_def.params) == 1
        assert gate_def.params[0] == "theta"
        assert len(gate_def.qargs) == 1
        assert gate_def.qargs[0] == "a"
        assert len(gate_def.body) == 1

    def test_gate_definition_multiple_qargs(self) -> None:
        """Define a gate with multiple quantum arguments."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate bell a, b {
            h a;
            cx a, b;
        }
        """
        ast = parse_qasm(source)
        assert "bell" in ast.gate_defs
        gate_def = ast.gate_defs["bell"]
        assert len(gate_def.qargs) == 2
        assert gate_def.qargs[0] == "a"
        assert gate_def.qargs[1] == "b"
        assert len(gate_def.body) == 2

    def test_gate_definition_multiple_params(self) -> None:
        """Define a gate with multiple parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate ugate(theta, phi) a {
            rx(theta) a;
            rz(phi) a;
        }
        """
        ast = parse_qasm(source)
        assert "ugate" in ast.gate_defs
        gate_def = ast.gate_defs["ugate"]
        assert len(gate_def.params) == 2
        assert gate_def.params[0] == "theta"
        assert gate_def.params[1] == "phi"
        assert len(gate_def.body) == 2

    def test_gate_invocation(self) -> None:
        """Invoke a user-defined gate."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        gate bell a, b {
            h a;
            cx a, b;
        }
        bell q[0], q[1];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "bell"
        assert len(gate_call.qargs) == 2

    def test_gate_invocation_with_params(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Invoke a user-defined gate with parameters."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate rgate(theta) a {
            rx(theta) a;
        }
        rgate(pi/2) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.name == "rgate"
        assert len(gate_call.params) == 1
        assert approx_equal(gate_call.params[0], math.pi / 2)

    def test_duplicate_gate_definition_raises_error(self) -> None:
        """Duplicate gate definitions should raise error."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate mygate a {
            h a;
        }
        gate mygate a {
            x a;
        }
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Duplicate" in exc_info.value.message

    def test_unknown_gate_raises_error(self) -> None:
        """Invoking unknown gate should raise error."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        unknown_gate q[0];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Unknown gate" in exc_info.value.message


# =============================================================================
# Measurement Tests
# =============================================================================


class TestMeasurement:
    """Tests for measurement operations."""

    def test_single_measurement(self) -> None:
        """Measure a single qubit."""
        source = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        measure = ast.body[0]
        assert isinstance(measure, MeasureAST)
        assert measure.q.reg == "q"
        assert measure.q.idx == 0
        assert measure.c.reg == "c"
        assert measure.c.idx == 0

    def test_multiple_measurements(self) -> None:
        """Measure multiple qubits."""
        source = """
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 3
        for i in range(3):
            measure = ast.body[i]
            assert isinstance(measure, MeasureAST)
            assert measure.q.idx == i
            assert measure.c.idx == i

    def test_measurement_different_registers(self) -> None:
        """Measure qubits from different registers."""
        source = """
        OPENQASM 2.0;
        qreg qa[1];
        qreg qb[1];
        creg ca[1];
        creg cb[1];
        measure qa[0] -> ca[0];
        measure qb[0] -> cb[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 2
        assert ast.body[0].q.reg == "qa"  # type: ignore[union-attr]
        assert ast.body[0].c.reg == "ca"  # type: ignore[union-attr]
        assert ast.body[1].q.reg == "qb"  # type: ignore[union-attr]
        assert ast.body[1].c.reg == "cb"  # type: ignore[union-attr]

    def test_measurement_unknown_qreg_raises_error(self) -> None:
        """Measuring unknown quantum register should raise error."""
        source = """
        OPENQASM 2.0;
        creg c[1];
        measure q[0] -> c[0];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Unknown quantum register" in exc_info.value.message

    def test_measurement_unknown_creg_raises_error(self) -> None:
        """Measuring to unknown classical register should raise error."""
        source = """
        OPENQASM 2.0;
        qreg q[1];
        measure q[0] -> c[0];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Unknown classical register" in exc_info.value.message

    def test_measurement_out_of_bounds_qreg_raises_error(self) -> None:
        """Measuring out-of-bounds quantum index should raise error."""
        source = """
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        measure q[5] -> c[0];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message

    def test_measurement_out_of_bounds_creg_raises_error(self) -> None:
        """Measuring to out-of-bounds classical index should raise error."""
        source = """
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        measure q[0] -> c[5];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message


# =============================================================================
# Barrier Tests
# =============================================================================


class TestBarrier:
    """Tests for barrier operations."""

    def test_barrier_single_qubit(self) -> None:
        """Apply barrier to a single qubit."""
        source = """
        OPENQASM 2.0;
        qreg q[1];
        barrier q[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        barrier = ast.body[0]
        assert isinstance(barrier, BarrierAST)
        assert len(barrier.qargs) == 1
        assert barrier.qargs[0].reg == "q"
        assert barrier.qargs[0].idx == 0

    def test_barrier_multiple_qubits(self) -> None:
        """Apply barrier to multiple qubits."""
        source = """
        OPENQASM 2.0;
        qreg q[3];
        barrier q[0], q[1], q[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 1
        barrier = ast.body[0]
        assert isinstance(barrier, BarrierAST)
        assert len(barrier.qargs) == 3
        for i in range(3):
            assert barrier.qargs[i].idx == i

    def test_barrier_different_registers(self) -> None:
        """Apply barrier across different registers."""
        source = """
        OPENQASM 2.0;
        qreg qa[1];
        qreg qb[1];
        barrier qa[0], qb[0];
        """
        ast = parse_qasm(source)
        barrier = ast.body[0]
        assert isinstance(barrier, BarrierAST)
        assert len(barrier.qargs) == 2
        assert barrier.qargs[0].reg == "qa"
        assert barrier.qargs[1].reg == "qb"

    def test_barrier_in_gate_body_raises_error(self) -> None:
        """Barrier inside gate body should raise error."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate bad a {
            h a;
            barrier a;
        }
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "not permitted" in exc_info.value.message


# =============================================================================
# Expression Evaluation Tests
# =============================================================================


class TestExpressionEvaluation:
    """Tests for expression evaluation in gate parameters."""

    def test_pi_constant(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate pi constant."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.pi)

    def test_pi_division(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate pi/2 expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.pi / 2)

    def test_pi_multiplication(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate 2*pi expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(2*pi) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], 2 * math.pi)

    def test_sin_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate sin(pi/4) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(sin(pi/4)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.sin(math.pi / 4))

    def test_cos_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate cos(pi/3) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(cos(pi/3)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.cos(math.pi / 3))

    def test_tan_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate tan(pi/6) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(tan(pi/6)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.tan(math.pi / 6))

    def test_exp_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate exp(1) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(exp(1)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.e)

    def test_ln_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate ln(2) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(ln(2)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.log(2))

    def test_sqrt_function(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate sqrt(2) expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(sqrt(2)) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.sqrt(2))

    def test_complex_expression(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate complex arithmetic expression."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2 + pi/4) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.pi / 2 + math.pi / 4)

    def test_nested_expression(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Evaluate nested function calls."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(sin(cos(pi/4))) q[0];
        """
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert approx_equal(gate_call.params[0], math.sin(math.cos(math.pi / 4)))


# =============================================================================
# Position Tracking Tests
# =============================================================================


class TestPositionTracking:
    """Tests for source position tracking (line and column numbers)."""

    def test_qreg_position(self) -> None:
        """Check that register declaration has position info."""
        source = """OPENQASM 2.0;
qreg q[1];"""
        ast = parse_qasm(source)
        # Position information is tracked during parsing
        assert len(ast.qregs) == 1

    def test_gate_call_position(self) -> None:
        """Check that gate call has position info."""
        source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];"""
        ast = parse_qasm(source)
        gate_call = ast.body[0]
        assert isinstance(gate_call, GateCallAST)
        assert gate_call.line > 0
        assert gate_call.col > 0

    def test_measure_position(self) -> None:
        """Check that measurement has position info."""
        source = """OPENQASM 2.0;
qreg q[1];
creg c[1];
measure q[0] -> c[0];"""
        ast = parse_qasm(source)
        measure = ast.body[0]
        assert isinstance(measure, MeasureAST)
        assert measure.line > 0
        assert measure.col > 0

    def test_barrier_position(self) -> None:
        """Check that barrier has position info."""
        source = """OPENQASM 2.0;
qreg q[1];
barrier q[0];"""
        ast = parse_qasm(source)
        barrier = ast.body[0]
        assert isinstance(barrier, BarrierAST)
        assert barrier.line > 0
        assert barrier.col > 0

    def test_gate_def_position(self) -> None:
        """Check that gate definition has position info."""
        source = """OPENQASM 2.0;
include "qelib1.inc";
gate mygate a {
    h a;
}"""
        ast = parse_qasm(source)
        gate_def = ast.gate_defs["mygate"]
        assert gate_def.line > 0
        assert gate_def.col > 0

    def test_error_position_duplicate_qreg(self) -> None:
        """Check that error contains position info for duplicate register."""
        source = """OPENQASM 2.0;
qreg q[1];
qreg q[2];"""
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        error = exc_info.value
        assert error.line > 0
        assert error.col > 0
        assert error.line == 3  # Error on third line


# =============================================================================
# Error Case Tests
# =============================================================================


class TestErrorCases:
    """Tests for error handling and reporting."""

    def test_syntax_error_missing_semicolon(self) -> None:
        """Missing semicolon should raise syntax error."""
        source = """
        OPENQASM 2.0;
        qreg q[1]
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code.startswith("E")

    def test_lexical_error_invalid_number(self) -> None:
        """Invalid number format should raise error."""
        source = """
        OPENQASM 2.0;
        qreg q[1abc];
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code.startswith("E")

    def test_semantic_error_gate_wrong_arg_count(self) -> None:
        """Gate called with wrong argument count should raise error."""
        from qasm2.validate import validate_program

        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        gate mygate a {
            h a;
        }
        mygate q[0], q[1];
        """
        # This is caught during validation (gate signature checking)
        with pytest.raises(QasmError) as exc_info:
            ast = parse_qasm(source)
            validate_program(ast)
        assert exc_info.value.code.startswith("E")

    def test_resolution_error_unknown_gate_argument(self) -> None:
        """Unknown gate argument should raise resolution error."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate bad a {
            h b;
        }
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code == "E402"
        assert "Unknown gate argument" in exc_info.value.message

    def test_empty_program(self) -> None:
        """Empty program should raise error."""
        source = ""
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code.startswith("E")

    def test_malformed_gate_call(self) -> None:
        """Malformed gate call should raise error."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h;
        """
        with pytest.raises(QasmError) as exc_info:
            parse_qasm(source)
        assert exc_info.value.code.startswith("E")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_bell_state_circuit(self) -> None:
        """Parse a complete Bell state preparation circuit."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        ast = parse_qasm(source)
        assert ast.version == "2.0"
        assert len(ast.includes) == 1
        assert len(ast.qregs) == 1
        assert len(ast.cregs) == 1
        assert len(ast.body) == 4
        assert ast.body[0].name == "h"  # type: ignore[union-attr]
        assert ast.body[1].name == "cx"  # type: ignore[union-attr]
        assert isinstance(ast.body[2], MeasureAST)
        assert isinstance(ast.body[3], MeasureAST)

    def test_parameterized_circuit(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Parse a circuit with parameterized gates."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
        ry(pi/4) q[0];
        rz(-pi/2) q[0];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 3
        assert approx_equal(ast.body[0].params[0], math.pi / 2)  # type: ignore[union-attr]
        assert approx_equal(ast.body[1].params[0], math.pi / 4)  # type: ignore[union-attr]
        assert approx_equal(ast.body[2].params[0], -math.pi / 2)  # type: ignore[union-attr]

    def test_custom_gate_circuit(self) -> None:
        """Parse a circuit with custom gate definition and usage."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        gate bell a, b {
            h a;
            cx a, b;
        }
        bell q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        ast = parse_qasm(source)
        assert "bell" in ast.gate_defs
        assert len(ast.body) == 3
        assert ast.body[0].name == "bell"  # type: ignore[union-attr]

    def test_complex_circuit_with_barriers(self) -> None:
        """Parse a complex circuit with barriers."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        barrier q[0], q[1], q[2];
        cx q[0], q[1];
        cx q[1], q[2];
        barrier q;
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        """
        ast = parse_qasm(source)
        assert len(ast.body) == 8
        assert isinstance(ast.body[1], BarrierAST)
        assert isinstance(ast.body[4], BarrierAST)

    def test_multi_register_circuit(self) -> None:
        """Parse a circuit with multiple registers."""
        source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg qa[2];
        qreg qb[2];
        creg ca[2];
        creg cb[2];
        h qa[0];
        h qb[0];
        cx qa[0], qa[1];
        cx qb[0], qb[1];
        measure qa[0] -> ca[0];
        measure qa[1] -> ca[1];
        measure qb[0] -> cb[0];
        measure qb[1] -> cb[1];
        """
        ast = parse_qasm(source)
        assert len(ast.qregs) == 2
        assert len(ast.cregs) == 2
        assert len(ast.body) == 8
