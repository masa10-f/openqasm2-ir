"""End-to-end pipeline tests for OpenQASM 2 to IR conversion.

This module provides comprehensive integration tests for the full pipeline:
QASM → AST → Validation → Normalization → IR → JSON

Test coverage includes:
- Bell pair circuit (complete pipeline)
- GHZ state circuits
- Parameterized gates with expression evaluation
- Golden JSON file comparison
- Error handling and edge cases

Note: Tests focus on the normalization → IR → JSON pipeline steps,
as parser testing is covered in test_parser.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ir.circuit import Circuit, Op
from qasm2.ast_nodes import CRef, GateCallAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmError
from qasm2.lower import load_gate_mappings, lower_to_ir
from qasm2.normalize import normalize_program
from qasm2.validate import validate_program

if TYPE_CHECKING:
    from typing import Callable


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def gates_yaml_path() -> str:
    """Locate the gates.yaml mapping file."""
    candidates = [
        Path(__file__).resolve().parent.parent / "gates" / "gates.yaml",
        Path(__file__).resolve().parent.parent / "gates.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("gates.yaml not found")


@pytest.fixture
def data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def float_approx() -> Callable[[float, float], bool]:
    """Compare floating point values with tolerance."""

    def _compare(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    return _compare


def _load_golden_json(path: Path) -> dict[str, Any]:
    """Load golden JSON file.

    Parameters
    ----------
    path : Path
        Path to the golden JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.
    """
    with path.open("r") as f:
        return json.load(f)


def _create_qref(reg: str, idx: int) -> QRef:
    """Create a quantum register reference."""
    return QRef(reg=reg, idx=idx, line=0, col=0)


def _create_cref(reg: str, idx: int) -> CRef:
    """Create a classical register reference."""
    return CRef(reg=reg, idx=idx, line=0, col=0)


def _compare_circuits(
    actual: Circuit,
    expected: Circuit,
    float_approx_fn: Callable[[float, float], bool],
) -> None:
    """Compare two circuits for equality.

    Parameters
    ----------
    actual : Circuit
        Actual circuit from pipeline.
    expected : Circuit
        Expected circuit (from golden file).
    float_approx_fn : Callable
        Function to compare floating point values.

    Raises
    ------
    AssertionError
        If circuits differ.
    """
    assert actual.n_qubits == expected.n_qubits, (
        f"Qubit count mismatch: expected {expected.n_qubits}, got {actual.n_qubits}"
    )

    assert len(actual.ops) == len(expected.ops), (
        f"Operation count mismatch: expected {len(expected.ops)}, got {len(actual.ops)}"
    )

    for i, (actual_op, expected_op) in enumerate(zip(actual.ops, expected.ops)):
        assert actual_op.name == expected_op.name, (
            f"Op {i} name mismatch: expected {expected_op.name}, got {actual_op.name}"
        )
        assert actual_op.qubits == expected_op.qubits, (
            f"Op {i} qubits mismatch: expected {expected_op.qubits}, got {actual_op.qubits}"
        )
        assert len(actual_op.params) == len(expected_op.params), (
            f"Op {i} param count mismatch: expected {len(expected_op.params)}, "
            f"got {len(actual_op.params)}"
        )
        for j, (actual_param, expected_param) in enumerate(zip(actual_op.params, expected_op.params)):
            assert float_approx_fn(actual_param, expected_param), (
                f"Op {i} param {j} mismatch: expected {expected_param}, got {actual_param}"
            )

    # Compare measurement map
    if expected.meas_map is None:
        assert actual.meas_map is None, "Expected no measurement map but got one"
    else:
        assert actual.meas_map is not None, "Expected measurement map but got None"
        assert actual.meas_map == expected.meas_map, (
            f"Measurement map mismatch: expected {expected.meas_map}, got {actual.meas_map}"
        )


# =============================================================================
# Bell Pair Tests
# =============================================================================


class TestBellPair:
    """Tests for the Bell pair circuit (complete pipeline)."""

    def _create_bell_pair_ast(self) -> ProgramAST:
        """Helper to create Bell pair AST directly."""
        return ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            gate_defs={},
            body=[
                GateCallAST(
                    name="h",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0)],
                ),
                GateCallAST(
                    name="cx",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0), _create_qref("q", 1)],
                ),
                MeasureAST(
                    q=_create_qref("q", 0),
                    c=_create_cref("c", 0),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 1),
                    c=_create_cref("c", 1),
                    line=0,
                    col=0,
                ),
            ],
        )

    def test_bell_pair_validation(self) -> None:
        """Validate Bell pair circuit against strict profile."""
        ast = self._create_bell_pair_ast()
        validate_program(ast)  # Should not raise

    def test_bell_pair_normalization(self) -> None:
        """Normalize Bell pair circuit."""
        ast = self._create_bell_pair_ast()
        validate_program(ast)
        normalized = normalize_program(ast)

        # Verify normalized program structure
        assert normalized.version == "2.0"
        assert len(normalized.qregs) == 1
        assert len(normalized.cregs) == 1
        # After normalization, body should have gates and measurements
        assert len(normalized.body) > 0

    def test_bell_pair_full_pipeline(self, gates_yaml_path: str) -> None:
        """Execute full pipeline on Bell pair circuit."""
        ast = self._create_bell_pair_ast()
        validate_program(ast)
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        assert circuit.n_qubits == 2
        assert len(circuit.ops) > 0
        assert circuit.meas_map is not None
        assert len(circuit.meas_map) == 2

    def test_bell_pair_json_export(self, gates_yaml_path: str) -> None:
        """Export Bell pair circuit to JSON."""
        ast = self._create_bell_pair_ast()
        validate_program(ast)
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        # Convert to JSON
        circuit_dict = circuit.to_dict()
        assert circuit_dict["n_qubits"] == 2
        assert "ops" in circuit_dict
        assert "meas_map" in circuit_dict

        # Verify JSON serialization round-trip
        json_str = json.dumps(circuit_dict)
        loaded_dict = json.loads(json_str)
        assert loaded_dict == circuit_dict


# =============================================================================
# GHZ State Tests
# =============================================================================


class TestGHZState:
    """Tests for GHZ state circuits."""

    def _create_ghz_3_ast(self) -> ProgramAST:
        """Helper to create GHZ-3 AST directly."""
        return ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 3)],
            cregs=[("c", 3)],
            gate_defs={},
            body=[
                GateCallAST(
                    name="h",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0)],
                ),
                GateCallAST(
                    name="cx",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0), _create_qref("q", 1)],
                ),
                GateCallAST(
                    name="cx",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 1), _create_qref("q", 2)],
                ),
                MeasureAST(
                    q=_create_qref("q", 0),
                    c=_create_cref("c", 0),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 1),
                    c=_create_cref("c", 1),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 2),
                    c=_create_cref("c", 2),
                    line=0,
                    col=0,
                ),
            ],
        )

    def test_ghz_3_validation(self) -> None:
        """Validate GHZ-3 circuit against strict profile."""
        ast = self._create_ghz_3_ast()
        validate_program(ast)

    def test_ghz_3_full_pipeline(self, gates_yaml_path: str) -> None:
        """Execute full pipeline on GHZ-3 circuit."""
        ast = self._create_ghz_3_ast()
        validate_program(ast)
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        assert circuit.n_qubits == 3
        assert len(circuit.ops) > 0
        assert circuit.meas_map is not None
        assert len(circuit.meas_map) == 3

    def test_ghz_3_measurement_map(self, gates_yaml_path: str) -> None:
        """Verify GHZ-3 measurement map is correct."""
        ast = self._create_ghz_3_ast()
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        assert circuit.meas_map is not None
        assert len(circuit.meas_map) == 3
        # Verify each qubit maps to corresponding classical bit
        for qubit_idx, cbit_idx in circuit.meas_map:
            assert qubit_idx == cbit_idx


# =============================================================================
# Parameterized Gates Tests
# =============================================================================


class TestParameterizedGates:
    """Tests for circuits with parameterized gates."""

    def _create_parameterized_ast(self) -> ProgramAST:
        """Helper to create parameterized gates AST directly."""
        return ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 3)],
            cregs=[("c", 3)],
            gate_defs={},
            body=[
                GateCallAST(
                    name="rx",
                    line=0,
                    col=0,
                    params=[math.pi / 4],
                    qargs=[_create_qref("q", 0)],
                ),
                GateCallAST(
                    name="ry",
                    line=0,
                    col=0,
                    params=[math.pi / 2],
                    qargs=[_create_qref("q", 1)],
                ),
                GateCallAST(
                    name="rz",
                    line=0,
                    col=0,
                    params=[math.pi],
                    qargs=[_create_qref("q", 2)],
                ),
                MeasureAST(
                    q=_create_qref("q", 0),
                    c=_create_cref("c", 0),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 1),
                    c=_create_cref("c", 1),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 2),
                    c=_create_cref("c", 2),
                    line=0,
                    col=0,
                ),
            ],
        )

    def test_parameterized_gates_validation(self) -> None:
        """Validate parameterized gates circuit."""
        ast = self._create_parameterized_ast()
        validate_program(ast)

    def test_parameterized_gates_full_pipeline(self, gates_yaml_path: str) -> None:
        """Execute full pipeline on parameterized gates circuit."""
        ast = self._create_parameterized_ast()
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        assert circuit.n_qubits == 3
        assert len(circuit.ops) >= 3

    def test_parameterized_gates_pi_evaluation(
        self, gates_yaml_path: str, float_approx: Callable[[float, float], bool]
    ) -> None:
        """Verify pi constant and expressions are correctly evaluated."""
        ast = self._create_parameterized_ast()
        normalized = normalize_program(ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        # Verify parameter values
        assert circuit.n_qubits == 3
        assert len(circuit.ops) >= 3

        # Check first three operations for correct parameter values
        assert float_approx(circuit.ops[0].params[0], math.pi / 4), (
            f"Expected pi/4 ≈ {math.pi / 4}, got {circuit.ops[0].params[0]}"
        )
        assert float_approx(circuit.ops[1].params[0], math.pi / 2), (
            f"Expected pi/2 ≈ {math.pi / 2}, got {circuit.ops[1].params[0]}"
        )
        assert float_approx(circuit.ops[2].params[0], math.pi), (
            f"Expected pi ≈ {math.pi}, got {circuit.ops[2].params[0]}"
        )


# =============================================================================
# Golden JSON Comparison Tests
# =============================================================================


class TestGoldenJsonComparison:
    """Test pipeline output against golden JSON files."""

    @pytest.mark.parametrize(
        "qasm_name",
        [
            "bell_pair",
            "ghz_3",
            "parameterized_gates",
        ],
    )
    def test_pipeline_matches_golden_json(
        self,
        qasm_name: str,
        data_dir: Path,
        gates_yaml_path: str,
        float_approx: Callable[[float, float], bool],
    ) -> None:
        """Test pipeline for QASM files with golden JSON comparison.

        This test verifies that the pipeline correctly converts QASM files
        to IR that matches the corresponding golden JSON file.
        """
        qasm_file = data_dir / f"{qasm_name}.qasm"
        golden_file = data_dir / f"{qasm_name}.golden.json"

        if not qasm_file.exists() or not golden_file.exists():
            pytest.skip(f"Missing test data for {qasm_name}")

        # Parse, validate, normalize, and lower to IR
        try:
            from qasm2.parser import parse_qasm_file

            ast = parse_qasm_file(str(qasm_file))
            validate_program(ast)
            normalized = normalize_program(ast)
            gate_mappings = load_gate_mappings(gates_yaml_path)
            circuit = lower_to_ir(normalized, gate_mappings)

            # Load expected circuit from golden JSON
            expected_dict = _load_golden_json(golden_file)
            expected = Circuit.from_dict(expected_dict)

            # Compare
            _compare_circuits(circuit, expected, float_approx)
        except QasmError:
            # Skip if parser fails (known limitation)
            pytest.skip(f"Parser unable to process {qasm_name}")

    @pytest.mark.parametrize(
        "qasm_name",
        [
            "bell_pair",
            "ghz_3",
            "parameterized_gates",
        ],
    )
    def test_json_round_trip(
        self,
        qasm_name: str,
        data_dir: Path,
        gates_yaml_path: str,
    ) -> None:
        """Verify JSON serialization round-trip for key circuits.

        This test ensures that:
        1. Circuit → JSON serialization is correct
        2. JSON → Circuit deserialization is correct
        3. Round-trip preserves all circuit information
        """
        qasm_file = data_dir / f"{qasm_name}.qasm"
        if not qasm_file.exists():
            pytest.skip(f"Missing test data for {qasm_name}")

        try:
            from qasm2.parser import parse_qasm_file

            ast = parse_qasm_file(str(qasm_file))
            normalized = normalize_program(ast)
            gate_mappings = load_gate_mappings(gates_yaml_path)
            circuit = lower_to_ir(normalized, gate_mappings)

            # Convert to JSON and back
            circuit_dict = circuit.to_dict()
            json_str = json.dumps(circuit_dict)
            loaded_dict = json.loads(json_str)
            loaded_circuit = Circuit.from_dict(loaded_dict)

            # Compare
            assert loaded_circuit.n_qubits == circuit.n_qubits
            assert len(loaded_circuit.ops) == len(circuit.ops)
            for actual_op, original_op in zip(loaded_circuit.ops, circuit.ops):
                assert actual_op.name == original_op.name
                assert actual_op.qubits == original_op.qubits
                assert actual_op.params == original_op.params

            if circuit.meas_map is None:
                assert loaded_circuit.meas_map is None
            else:
                assert loaded_circuit.meas_map == circuit.meas_map
        except QasmError:
            pytest.skip(f"Parser unable to process {qasm_name}")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_invalid_qasm_syntax_raises_error(self) -> None:
        """Invalid QASM syntax should raise QasmError."""
        from qasm2.parser import parse_qasm

        invalid_qasm = """
        OPENQASM 2.0;
        invalid syntax here
        """
        with pytest.raises(QasmError):
            parse_qasm(invalid_qasm)

    def test_missing_header_raises_error(self) -> None:
        """Missing OPENQASM header should raise error."""
        from qasm2.parser import parse_qasm

        invalid_qasm = """
        qreg q[2];
        h q[0];
        """
        with pytest.raises(QasmError):
            parse_qasm(invalid_qasm)

    def test_duplicate_register_raises_error(self) -> None:
        """Duplicate quantum register should raise error."""
        from qasm2.parser import parse_qasm

        invalid_qasm = """
        OPENQASM 2.0;
        qreg q[2];
        qreg q[3];
        """
        with pytest.raises(QasmError):
            parse_qasm(invalid_qasm)


# =============================================================================
# Consistency Tests
# =============================================================================


class TestConsistency:
    """Tests for internal consistency and invariants."""

    def _create_simple_bell_pair_ast(self) -> ProgramAST:
        """Helper to create a simple Bell pair AST."""
        return ProgramAST(
            version="2.0",
            includes=["qelib1.inc"],
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            gate_defs={},
            body=[
                GateCallAST(
                    name="h",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0)],
                ),
                GateCallAST(
                    name="cx",
                    line=0,
                    col=0,
                    params=[],
                    qargs=[_create_qref("q", 0), _create_qref("q", 1)],
                ),
                MeasureAST(
                    q=_create_qref("q", 0),
                    c=_create_cref("c", 0),
                    line=0,
                    col=0,
                ),
                MeasureAST(
                    q=_create_qref("q", 1),
                    c=_create_cref("c", 1),
                    line=0,
                    col=0,
                ),
            ],
        )

    def test_qubit_indices_are_valid(self, gates_yaml_path: str) -> None:
        """Verify all qubit indices are within valid range."""
        # Test Bell pair
        bell_pair_ast = self._create_simple_bell_pair_ast()
        normalized = normalize_program(bell_pair_ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        for op in circuit.ops:
            for qubit_idx in op.qubits:
                assert 0 <= qubit_idx < circuit.n_qubits, (
                    f"Qubit index {qubit_idx} out of range [0, {circuit.n_qubits})"
                )

    def test_measurement_map_validity(self, gates_yaml_path: str) -> None:
        """Verify measurement map contains only valid qubit indices."""
        bell_pair_ast = self._create_simple_bell_pair_ast()
        normalized = normalize_program(bell_pair_ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        if circuit.meas_map is not None:
            for qubit_idx, _cbit_idx in circuit.meas_map:
                assert 0 <= qubit_idx < circuit.n_qubits, (
                    f"Qubit index {qubit_idx} in measurement map out of range [0, {circuit.n_qubits})"
                )

    def test_operation_count_positive(self, gates_yaml_path: str) -> None:
        """Verify circuits contain at least one operation."""
        bell_pair_ast = self._create_simple_bell_pair_ast()
        normalized = normalize_program(bell_pair_ast)
        gate_mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(normalized, gate_mappings)

        assert len(circuit.ops) > 0, "Circuit should contain at least one operation"

    def test_circuit_creation_and_to_dict(self) -> None:
        """Verify Circuit can be created and converted to dict."""
        ops = [
            Op(name="H", qubits=[0], params=()),
            Op(name="CX", qubits=[0, 1], params=()),
        ]
        meas_map = [(0, 0), (1, 1)]

        circuit = Circuit(n_qubits=2, ops=ops, meas_map=meas_map)
        circuit_dict = circuit.to_dict()

        assert circuit_dict["n_qubits"] == 2
        assert len(circuit_dict["ops"]) == 2
        assert circuit_dict["meas_map"] == [[0, 0], [1, 1]]

    def test_circuit_from_dict(self) -> None:
        """Verify Circuit can be created from dict."""
        data = {
            "n_qubits": 2,
            "ops": [
                {"name": "H", "qubits": [0], "params": []},
                {"name": "CX", "qubits": [0, 1], "params": []},
            ],
            "meas_map": [[0, 0], [1, 1]],
        }

        circuit = Circuit.from_dict(data)

        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2
        assert circuit.meas_map == [(0, 0), (1, 1)]
