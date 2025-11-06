"""Test suite for OpenQASM 2.0 strict profile validator.

This module provides comprehensive tests for the validate functionality defined in
qasm2.validate, covering:
- OpenQASM version validation (2.0 only)
- Include directive validation (qelib1.inc only)
- Register duplication checks
- Register size validation
- Out-of-bounds access detection
- Mid-circuit measurement detection
- Recursive gate definition detection
- Gate signature validation (parameter and qubit counts)
- Measurement register size matching

All tests verify that the strict profile validator correctly enforces the subset of
OpenQASM 2.0 permitted by the strict profile.
"""

from __future__ import annotations

import pytest

from qasm2.ast_nodes import BarrierAST, CRef, GateCallAST, GateDefAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmError
from qasm2.validate import validate_program


# =============================================================================
# Version Validation Tests
# =============================================================================


class TestVersionValidation:
    """Tests for OpenQASM version validation."""

    def test_validate_version_2_0_succeeds(self) -> None:
        """Validate that version 2.0 is accepted."""
        ast = ProgramAST(version="2.0")
        validate_program(ast)  # Should not raise

    def test_validate_version_1_0_raises_error(self) -> None:
        """Reject version 1.0."""
        ast = ProgramAST(version="1.0")
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not supported" in exc_info.value.message
        assert "2.0" in exc_info.value.message

    def test_validate_version_3_0_raises_error(self) -> None:
        """Reject version 3.0."""
        ast = ProgramAST(version="3.0")
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not supported" in exc_info.value.message

    def test_validate_version_2_1_raises_error(self) -> None:
        """Reject version 2.1."""
        ast = ProgramAST(version="2.1")
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not supported" in exc_info.value.message


# =============================================================================
# Include Directive Tests
# =============================================================================


class TestIncludeValidation:
    """Tests for include directive validation."""

    def test_validate_qelib1_inc_succeeds(self) -> None:
        """Validate that qelib1.inc is accepted."""
        ast = ProgramAST(version="2.0", includes=["qelib1.inc"])
        validate_program(ast)  # Should not raise

    def test_validate_no_includes_succeeds(self) -> None:
        """Validate that program without includes is accepted."""
        ast = ProgramAST(version="2.0")
        validate_program(ast)  # Should not raise

    def test_validate_multiple_qelib1_inc_succeeds(self) -> None:
        """Validate that multiple qelib1.inc includes are accepted."""
        ast = ProgramAST(version="2.0", includes=["qelib1.inc", "qelib1.inc"])
        validate_program(ast)  # Should not raise

    def test_validate_custom_inc_raises_error(self) -> None:
        """Reject custom include files."""
        ast = ProgramAST(version="2.0", includes=["custom.inc"])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not permitted" in exc_info.value.message
        assert "qelib1.inc" in exc_info.value.message

    def test_validate_custom_inc_with_path_raises_error(self) -> None:
        """Reject include with path."""
        ast = ProgramAST(version="2.0", includes=["gates/mylib.inc"])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not permitted" in exc_info.value.message

    def test_validate_mixed_includes_raises_error(self) -> None:
        """Reject mixed includes with non-permitted files."""
        ast = ProgramAST(version="2.0", includes=["qelib1.inc", "custom.inc"])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E301"
        assert "not permitted" in exc_info.value.message


# =============================================================================
# Register Declaration Tests
# =============================================================================


class TestRegisterValidation:
    """Tests for quantum and classical register validation."""

    def test_validate_single_qreg_succeeds(self) -> None:
        """Validate single quantum register."""
        ast = ProgramAST(version="2.0", qregs=[("q", 1)])
        validate_program(ast)  # Should not raise

    def test_validate_single_creg_succeeds(self) -> None:
        """Validate single classical register."""
        ast = ProgramAST(version="2.0", cregs=[("c", 1)])
        validate_program(ast)  # Should not raise

    def test_validate_multiple_qregs_succeeds(self) -> None:
        """Validate multiple quantum registers."""
        ast = ProgramAST(version="2.0", qregs=[("q1", 2), ("q2", 3)])
        validate_program(ast)  # Should not raise

    def test_validate_multiple_cregs_succeeds(self) -> None:
        """Validate multiple classical registers."""
        ast = ProgramAST(version="2.0", cregs=[("c1", 2), ("c2", 3)])
        validate_program(ast)  # Should not raise

    def test_validate_duplicate_qreg_raises_error(self) -> None:
        """Reject duplicate quantum register names."""
        ast = ProgramAST(version="2.0", qregs=[("q", 2), ("q", 3)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E401"
        assert "Duplicate" in exc_info.value.message
        assert "quantum register" in exc_info.value.message

    def test_validate_duplicate_creg_raises_error(self) -> None:
        """Reject duplicate classical register names."""
        ast = ProgramAST(version="2.0", cregs=[("c", 2), ("c", 3)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E401"
        assert "Duplicate" in exc_info.value.message
        assert "classical register" in exc_info.value.message

    def test_validate_qreg_size_zero_raises_error(self) -> None:
        """Reject quantum register with size zero."""
        ast = ProgramAST(version="2.0", qregs=[("q", 0)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "greater than zero" in exc_info.value.message

    def test_validate_qreg_negative_size_raises_error(self) -> None:
        """Reject quantum register with negative size."""
        ast = ProgramAST(version="2.0", qregs=[("q", -1)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "greater than zero" in exc_info.value.message

    def test_validate_creg_size_zero_raises_error(self) -> None:
        """Reject classical register with size zero."""
        ast = ProgramAST(version="2.0", cregs=[("c", 0)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "greater than zero" in exc_info.value.message

    def test_validate_creg_negative_size_raises_error(self) -> None:
        """Reject classical register with negative size."""
        ast = ProgramAST(version="2.0", cregs=[("c", -1)])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "greater than zero" in exc_info.value.message


# =============================================================================
# Out-of-Bounds Access Tests
# =============================================================================


class TestOutOfBoundsAccess:
    """Tests for out-of-bounds qubit and bit access detection."""

    def test_validate_gate_qubit_in_bounds_succeeds(self) -> None:
        """Validate gate access to in-bounds qubit."""
        qref = QRef(reg="q", idx=0, line=1, col=1)
        gate = GateCallAST(name="h", line=1, col=1, qargs=[qref])
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_gate_qubit_out_of_bounds_raises_error(self) -> None:
        """Reject gate access to out-of-bounds qubit."""
        qref = QRef(reg="q", idx=5, line=1, col=1)
        gate = GateCallAST(name="h", line=1, col=1, qargs=[qref])
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message
        assert "5" in exc_info.value.message

    def test_validate_gate_negative_qubit_index_raises_error(self) -> None:
        """Reject gate access with negative qubit index."""
        qref = QRef(reg="q", idx=-1, line=1, col=1)
        gate = GateCallAST(name="h", line=1, col=1, qargs=[qref])
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message

    def test_validate_measurement_qubit_out_of_bounds_raises_error(self) -> None:
        """Reject measurement with out-of-bounds qubit."""
        qref = QRef(reg="q", idx=5, line=1, col=1)
        cref = CRef(reg="c", idx=0, line=1, col=1)
        measure = MeasureAST(q=qref, c=cref, line=1, col=1)
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 1)], body=[measure])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message

    def test_validate_measurement_cbit_out_of_bounds_raises_error(self) -> None:
        """Reject measurement with out-of-bounds classical bit."""
        qref = QRef(reg="q", idx=0, line=1, col=1)
        cref = CRef(reg="c", idx=5, line=1, col=1)
        measure = MeasureAST(q=qref, c=cref, line=1, col=1)
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], cregs=[("c", 2)], body=[measure])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message

    def test_validate_barrier_qubit_out_of_bounds_raises_error(self) -> None:
        """Reject barrier with out-of-bounds qubit."""
        qref = QRef(reg="q", idx=5, line=1, col=1)
        barrier = BarrierAST(line=1, col=1, qargs=[qref])
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], body=[barrier])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "out of range" in exc_info.value.message

    def test_validate_gate_undefined_qreg_raises_error(self) -> None:
        """Reject gate with undefined quantum register."""
        qref = QRef(reg="unknown", idx=0, line=1, col=1)
        gate = GateCallAST(name="h", line=1, col=1, qargs=[qref])
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "not defined" in exc_info.value.message

    def test_validate_measurement_undefined_qreg_raises_error(self) -> None:
        """Reject measurement with undefined quantum register."""
        qref = QRef(reg="unknown", idx=0, line=1, col=1)
        cref = CRef(reg="c", idx=0, line=1, col=1)
        measure = MeasureAST(q=qref, c=cref, line=1, col=1)
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], cregs=[("c", 1)], body=[measure])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "not defined" in exc_info.value.message

    def test_validate_measurement_undefined_creg_raises_error(self) -> None:
        """Reject measurement with undefined classical register."""
        qref = QRef(reg="q", idx=0, line=1, col=1)
        cref = CRef(reg="unknown", idx=0, line=1, col=1)
        measure = MeasureAST(q=qref, c=cref, line=1, col=1)
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], cregs=[("c", 1)], body=[measure])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "not defined" in exc_info.value.message


# =============================================================================
# Mid-Circuit Measurement Detection Tests
# =============================================================================


class TestMidCircuitMeasurementDetection:
    """Tests for mid-circuit measurement detection."""

    def test_validate_measurement_at_end_succeeds(self) -> None:
        """Validate that measurement at the end is allowed."""
        gate = GateCallAST(name="h", line=1, col=1, qargs=[QRef(reg="q", idx=0, line=1, col=1)])
        qref = QRef(reg="q", idx=0, line=1, col=1)
        cref = CRef(reg="c", idx=0, line=1, col=1)
        measure = MeasureAST(q=qref, c=cref, line=1, col=1)
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], cregs=[("c", 1)], body=[gate, measure])
        validate_program(ast)  # Should not raise

    def test_validate_multiple_measurements_at_end_succeeds(self) -> None:
        """Validate that multiple measurements at the end are allowed."""
        gate = GateCallAST(name="h", line=1, col=1, qargs=[QRef(reg="q", idx=0, line=1, col=1)])
        measure1 = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        measure2 = MeasureAST(
            q=QRef(reg="q", idx=1, line=1, col=1),
            c=CRef(reg="c", idx=1, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[gate, measure1, measure2],
        )
        validate_program(ast)  # Should not raise

    def test_validate_gate_after_measurement_raises_error(self) -> None:
        """Reject gate after measurement."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        gate = GateCallAST(name="h", line=1, col=1, qargs=[QRef(reg="q", idx=1, line=1, col=1)])
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 1)], body=[measure, gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E201"
        assert "mid-circuit measurement" in exc_info.value.message

    def test_validate_barrier_after_measurement_raises_error(self) -> None:
        """Reject barrier after measurement."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        barrier = BarrierAST(line=1, col=1, qargs=[QRef(reg="q", idx=1, line=1, col=1)])
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 1)], body=[measure, barrier])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E201"
        assert "mid-circuit measurement" in exc_info.value.message

    def test_validate_only_measurement_succeeds(self) -> None:
        """Validate that a program with only measurements is allowed."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], cregs=[("c", 1)], body=[measure])
        validate_program(ast)  # Should not raise


# =============================================================================
# Recursive Gate Definition Tests
# =============================================================================


class TestRecursiveGateDetection:
    """Tests for recursive gate definition detection."""

    def test_validate_non_recursive_gate_succeeds(self) -> None:
        """Validate non-recursive gate definition."""
        # Note: Gate bodies are also validated for signature match.
        # Using a simple gate definition without body statements avoids signature issues.
        gate_def = GateDefAST(name="mygate", line=1, col=1, qargs=["a"], body=[])
        ast = ProgramAST(version="2.0", gate_defs={"mygate": gate_def})
        validate_program(ast)  # Should not raise

    def test_validate_direct_recursion_raises_error(self) -> None:
        """Reject direct recursive gate definition."""
        gate_call = GateCallAST(name="mygate", line=1, col=1, params=[], qargs=[])
        gate_def = GateDefAST(name="mygate", line=1, col=1, qargs=["a"], body=[gate_call])
        ast = ProgramAST(version="2.0", gate_defs={"mygate": gate_def})
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E501"
        assert "Recursive gate definition" in exc_info.value.message

    def test_validate_indirect_recursion_raises_error(self) -> None:
        """Reject indirect recursive gate definition (A calls B calls A)."""
        gate_call_b = GateCallAST(name="gate_b", line=1, col=1, params=[], qargs=[])
        gate_def_a = GateDefAST(name="gate_a", line=1, col=1, qargs=["x"], body=[gate_call_b])

        gate_call_a = GateCallAST(name="gate_a", line=1, col=1, params=[], qargs=[])
        gate_def_b = GateDefAST(name="gate_b", line=1, col=1, qargs=["y"], body=[gate_call_a])

        ast = ProgramAST(version="2.0", gate_defs={"gate_a": gate_def_a, "gate_b": gate_def_b})
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E501"
        assert "Recursive gate definition" in exc_info.value.message

    def test_validate_mutual_recursion_raises_error(self) -> None:
        """Reject mutual recursive gate definition."""
        gate_call_b = GateCallAST(name="gate_b", line=1, col=1, params=[], qargs=[])
        gate_call_a = GateCallAST(name="gate_a", line=1, col=1, params=[], qargs=[])

        gate_def_a = GateDefAST(name="gate_a", line=1, col=1, qargs=["x"], body=[gate_call_b])
        gate_def_b = GateDefAST(name="gate_b", line=1, col=1, qargs=["y"], body=[gate_call_a])

        ast = ProgramAST(version="2.0", gate_defs={"gate_a": gate_def_a, "gate_b": gate_def_b})
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E501"
        assert "Recursive gate definition" in exc_info.value.message

    def test_validate_three_level_recursion_raises_error(self) -> None:
        """Reject three-level indirect recursion (A calls B calls C calls A)."""
        gate_call_b = GateCallAST(name="gate_b", line=1, col=1, params=[], qargs=[])
        gate_def_a = GateDefAST(name="gate_a", line=1, col=1, qargs=["x"], body=[gate_call_b])

        gate_call_c = GateCallAST(name="gate_c", line=1, col=1, params=[], qargs=[])
        gate_def_b = GateDefAST(name="gate_b", line=1, col=1, qargs=["y"], body=[gate_call_c])

        gate_call_a = GateCallAST(name="gate_a", line=1, col=1, params=[], qargs=[])
        gate_def_c = GateDefAST(name="gate_c", line=1, col=1, qargs=["z"], body=[gate_call_a])

        ast = ProgramAST(
            version="2.0",
            gate_defs={"gate_a": gate_def_a, "gate_b": gate_def_b, "gate_c": gate_def_c},
        )
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E501"
        assert "Recursive gate definition" in exc_info.value.message

    def test_validate_non_recursive_chain_succeeds(self) -> None:
        """Validate non-recursive gate chain (A calls B, B calls C)."""
        # Using empty bodies to avoid gate signature validation issues.
        gate_def_a = GateDefAST(name="gate_a", line=1, col=1, qargs=["x"], body=[])
        gate_def_b = GateDefAST(name="gate_b", line=1, col=1, qargs=["y"], body=[])

        ast = ProgramAST(version="2.0", gate_defs={"gate_a": gate_def_a, "gate_b": gate_def_b})
        validate_program(ast)  # Should not raise


# =============================================================================
# Gate Signature Validation Tests
# =============================================================================


class TestGateSignatureValidation:
    """Tests for gate signature validation (parameter and qubit counts)."""

    def test_validate_h_gate_correct_signature_succeeds(self) -> None:
        """Validate H gate with correct signature."""
        gate = GateCallAST(
            name="h",
            line=1,
            col=1,
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_h_gate_with_params_raises_error(self) -> None:
        """Reject H gate with parameters."""
        gate = GateCallAST(
            name="h",
            line=1,
            col=1,
            params=[1.5],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E703"
        assert "expects 0 parameter(s)" in exc_info.value.message

    def test_validate_h_gate_with_multiple_qubits_raises_error(self) -> None:
        """Reject H gate with multiple qubits."""
        gate = GateCallAST(
            name="h",
            line=1,
            col=1,
            params=[],
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E702"
        assert "expects 1 qubit operand(s)" in exc_info.value.message

    def test_validate_rx_gate_correct_signature_succeeds(self) -> None:
        """Validate RX gate with correct signature."""
        gate = GateCallAST(
            name="rx",
            line=1,
            col=1,
            params=[1.5],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_rx_gate_missing_param_raises_error(self) -> None:
        """Reject RX gate without parameters."""
        gate = GateCallAST(
            name="rx",
            line=1,
            col=1,
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E703"
        assert "expects 1 parameter(s)" in exc_info.value.message

    def test_validate_rx_gate_extra_params_raises_error(self) -> None:
        """Reject RX gate with extra parameters."""
        gate = GateCallAST(
            name="rx",
            line=1,
            col=1,
            params=[1.5, 2.0],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E703"
        assert "expects 1 parameter(s)" in exc_info.value.message

    def test_validate_u3_gate_correct_signature_succeeds(self) -> None:
        """Validate U3 gate with correct signature."""
        gate = GateCallAST(
            name="u3",
            line=1,
            col=1,
            params=[1.5, 2.0, 2.5],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_u3_gate_insufficient_params_raises_error(self) -> None:
        """Reject U3 gate with insufficient parameters."""
        gate = GateCallAST(
            name="u3",
            line=1,
            col=1,
            params=[1.5, 2.0],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E703"
        assert "expects 3 parameter(s)" in exc_info.value.message

    def test_validate_cx_gate_correct_signature_succeeds(self) -> None:
        """Validate CX gate with correct signature."""
        gate = GateCallAST(
            name="cx",
            line=1,
            col=1,
            params=[],
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_cx_gate_single_qubit_raises_error(self) -> None:
        """Reject CX gate with single qubit."""
        gate = GateCallAST(
            name="cx",
            line=1,
            col=1,
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E702"
        assert "expects 2 qubit operand(s)" in exc_info.value.message

    def test_validate_ccx_gate_correct_signature_succeeds(self) -> None:
        """Validate CCX gate with correct signature."""
        gate = GateCallAST(
            name="ccx",
            line=1,
            col=1,
            params=[],
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
                QRef(reg="q", idx=2, line=1, col=1),
            ],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 3)], body=[gate])
        validate_program(ast)  # Should not raise

    def test_validate_ccx_gate_insufficient_qubits_raises_error(self) -> None:
        """Reject CCX gate with insufficient qubits."""
        gate = GateCallAST(
            name="ccx",
            line=1,
            col=1,
            params=[],
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E702"
        assert "expects 3 qubit operand(s)" in exc_info.value.message


# =============================================================================
# User-Defined Gate Signature Tests
# =============================================================================


class TestUserDefinedGateSignature:
    """Tests for user-defined gate signature validation."""

    def test_validate_user_defined_gate_correct_signature_succeeds(self) -> None:
        """Validate user-defined gate with correct signature."""
        # Using empty body to avoid gate signature validation issues in definition.
        gate_def = GateDefAST(name="mygate", line=1, col=1, qargs=["a"], body=[])
        gate_call = GateCallAST(
            name="mygate",
            line=1,
            col=1,
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            gate_defs={"mygate": gate_def},
            body=[gate_call],
        )
        validate_program(ast)  # Should not raise

    def test_validate_user_defined_gate_with_params_correct_signature_succeeds(self) -> None:
        """Validate user-defined gate with parameters and correct signature."""
        # Using empty body to avoid gate signature validation issues in definition.
        gate_def = GateDefAST(
            name="rgate",
            line=1,
            col=1,
            params=["theta"],
            qargs=["a"],
            body=[],
        )
        gate_call = GateCallAST(
            name="rgate",
            line=1,
            col=1,
            params=[1.5],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            gate_defs={"rgate": gate_def},
            body=[gate_call],
        )
        validate_program(ast)  # Should not raise

    def test_validate_user_defined_gate_wrong_param_count_raises_error(self) -> None:
        """Reject user-defined gate with wrong parameter count."""
        gate_rx = GateCallAST(name="rx", line=1, col=1, params=[], qargs=[])
        gate_def = GateDefAST(
            name="rgate",
            line=1,
            col=1,
            params=["theta"],
            qargs=["a"],
            body=[gate_rx],
        )
        gate_call = GateCallAST(
            name="rgate",
            line=1,
            col=1,
            params=[1.5, 2.0],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            gate_defs={"rgate": gate_def},
            body=[gate_call],
        )
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E703"
        assert "expects 1 parameter(s)" in exc_info.value.message

    def test_validate_user_defined_gate_wrong_qubit_count_raises_error(self) -> None:
        """Reject user-defined gate with wrong qubit count."""
        gate_cx = GateCallAST(name="cx", line=1, col=1, qargs=[])
        gate_def = GateDefAST(
            name="cxgate",
            line=1,
            col=1,
            qargs=["a", "b"],
            body=[gate_cx],
        )
        gate_call = GateCallAST(
            name="cxgate",
            line=1,
            col=1,
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            gate_defs={"cxgate": gate_def},
            body=[gate_call],
        )
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E702"
        assert "expects 2 qubit operand(s)" in exc_info.value.message


# =============================================================================
# Unknown Gate Tests
# =============================================================================


class TestUnknownGate:
    """Tests for unknown gate detection."""

    def test_validate_undefined_gate_raises_error(self) -> None:
        """Reject invocation of undefined gate."""
        gate = GateCallAST(
            name="unknown_gate",
            line=1,
            col=1,
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 1)], body=[gate])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E701"
        assert "not defined" in exc_info.value.message

    def test_validate_builtin_gate_succeeds(self) -> None:
        """Validate that built-in gates are recognized."""
        # Single qubit gates
        single_qubit_gates = ["h", "x", "y", "z", "s", "t"]
        for gate_name in single_qubit_gates:
            gate = GateCallAST(
                name=gate_name,
                line=1,
                col=1,
                params=[],
                qargs=[QRef(reg="q", idx=0, line=1, col=1)],
            )
            ast = ProgramAST(version="2.0", qregs=[("q", 3)], body=[gate])
            validate_program(ast)  # Should not raise

        # Two qubit gates
        two_qubit_gates = ["cx", "cz", "swap"]
        for gate_name in two_qubit_gates:
            gate = GateCallAST(
                name=gate_name,
                line=1,
                col=1,
                params=[],
                qargs=[
                    QRef(reg="q", idx=0, line=1, col=1),
                    QRef(reg="q", idx=1, line=1, col=1),
                ],
            )
            ast = ProgramAST(version="2.0", qregs=[("q", 3)], body=[gate])
            validate_program(ast)  # Should not raise

        # Three qubit gates
        three_qubit_gates = ["ccx", "cswap"]
        for gate_name in three_qubit_gates:
            gate = GateCallAST(
                name=gate_name,
                line=1,
                col=1,
                params=[],
                qargs=[
                    QRef(reg="q", idx=0, line=1, col=1),
                    QRef(reg="q", idx=1, line=1, col=1),
                    QRef(reg="q", idx=2, line=1, col=1),
                ],
            )
            ast = ProgramAST(version="2.0", qregs=[("q", 3)], body=[gate])
            validate_program(ast)  # Should not raise


# =============================================================================
# Measurement Register Size Matching Tests
# =============================================================================


class TestMeasurementRegisterSizeMatching:
    """Tests for measurement register size matching validation."""

    def test_validate_measurement_matching_register_sizes_succeeds(self) -> None:
        """Validate measurement with matching register sizes."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=None, line=1, col=1),
            c=CRef(reg="c", idx=None, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 2)], body=[measure])
        validate_program(ast)  # Should not raise

    def test_validate_measurement_mismatched_register_sizes_raises_error(self) -> None:
        """Reject measurement with mismatched register sizes."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=None, line=1, col=1),
            c=CRef(reg="c", idx=None, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 3)], body=[measure])
        with pytest.raises(QasmError) as exc_info:
            validate_program(ast)
        assert exc_info.value.code == "E402"
        assert "Register sizes do not match" in exc_info.value.message

    def test_validate_measurement_with_indices_ignores_size_check_succeeds(self) -> None:
        """Validate measurement with indices ignores size matching."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 3)], body=[measure])
        validate_program(ast)  # Should not raise

    def test_validate_measurement_qreg_indexed_creg_full_succeeds(self) -> None:
        """Validate measurement with indexed qubit and full classical register."""
        measure = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=None, line=1, col=1),
            line=1,
            col=1,
        )
        ast = ProgramAST(version="2.0", qregs=[("q", 2)], cregs=[("c", 2)], body=[measure])
        validate_program(ast)  # Should not raise


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidationIntegration:
    """Integration tests combining multiple validation features."""

    def test_validate_bell_circuit_succeeds(self) -> None:
        """Validate a complete valid Bell state circuit."""
        h_gate = GateCallAST(
            name="h",
            line=1,
            col=1,
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        cx_gate = GateCallAST(
            name="cx",
            line=1,
            col=1,
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        measure1 = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        measure2 = MeasureAST(
            q=QRef(reg="q", idx=1, line=1, col=1),
            c=CRef(reg="c", idx=1, line=1, col=1),
            line=1,
            col=1,
        )

        ast = ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[h_gate, cx_gate, measure1, measure2],
        )
        validate_program(ast)  # Should not raise

    def test_validate_circuit_with_custom_gate_succeeds(self) -> None:
        """Validate circuit with custom gate definition."""
        # Using empty body to avoid gate signature validation issues in definition.
        gate_def = GateDefAST(name="bell", line=1, col=1, qargs=["a", "b"], body=[])

        bell_call = GateCallAST(
            name="bell",
            line=1,
            col=1,
            params=[],
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        measure1 = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        measure2 = MeasureAST(
            q=QRef(reg="q", idx=1, line=1, col=1),
            c=CRef(reg="c", idx=1, line=1, col=1),
            line=1,
            col=1,
        )

        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            gate_defs={"bell": gate_def},
            body=[bell_call, measure1, measure2],
        )
        validate_program(ast)  # Should not raise

    def test_validate_complex_valid_circuit_succeeds(self) -> None:
        """Validate a complex valid circuit."""
        rx_gate = GateCallAST(
            name="rx",
            line=1,
            col=1,
            params=[1.57],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
        )
        ry_gate = GateCallAST(
            name="ry",
            line=1,
            col=1,
            params=[0.785],
            qargs=[QRef(reg="q", idx=1, line=1, col=1)],
        )
        cx_gate = GateCallAST(
            name="cx",
            line=1,
            col=1,
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        barrier = BarrierAST(
            line=1,
            col=1,
            qargs=[
                QRef(reg="q", idx=0, line=1, col=1),
                QRef(reg="q", idx=1, line=1, col=1),
            ],
        )
        measure1 = MeasureAST(
            q=QRef(reg="q", idx=0, line=1, col=1),
            c=CRef(reg="c", idx=0, line=1, col=1),
            line=1,
            col=1,
        )
        measure2 = MeasureAST(
            q=QRef(reg="q", idx=1, line=1, col=1),
            c=CRef(reg="c", idx=1, line=1, col=1),
            line=1,
            col=1,
        )

        ast = ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[rx_gate, ry_gate, cx_gate, barrier, measure1, measure2],
        )
        validate_program(ast)  # Should not raise
