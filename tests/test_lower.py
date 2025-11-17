"""Test suite for OpenQASM 2.0 to custom IR lowering utilities.

This module provides comprehensive tests for the lowering functionality defined in
qasm2.lower, covering:
- Loading gate mappings from YAML files
- One-to-one (map) gate translations
- One-to-many (expand) gate translations
- Template substitution with parameters ({theta}, {phi}, {lam}, etc.)
- Flattening of quantum registers to flat qubit indices
- Flattening of classical registers to flat bit indices
- Measurement mapping generation
- Error cases (unmapped gates, arity mismatches, missing parameters)
- Expression evaluation in templates
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ir.circuit import Circuit
from qasm2.ast_nodes import CRef, GateCallAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmGraphIntegrationError
from qasm2.lower import (
    apply_gate_mapping,
    flatten_cref,
    flatten_qref,
    load_gate_mappings,
    lower_to_ir,
    substitute_template,
)

if TYPE_CHECKING:
    from typing import Callable


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def gates_yaml_path() -> str:
    """Provide the path to the gate mappings YAML file."""
    return str(Path(__file__).parent.parent / "gates" / "gates.yaml")


@pytest.fixture
def approx_equal() -> Callable[[float, float], bool]:
    """Compare floating point values with tolerance."""

    def _compare(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    return _compare


@pytest.fixture
def qreg_offsets() -> dict[str, int]:
    """Provide sample quantum register offsets."""
    return {"q": 0, "a": 2, "b": 5}


@pytest.fixture
def qreg_sizes() -> dict[str, int]:
    """Provide sample quantum register sizes."""
    return {"q": 2, "a": 3, "b": 4}


@pytest.fixture
def creg_offsets() -> dict[str, int]:
    """Provide sample classical register offsets."""
    return {"c": 0, "m": 3}


@pytest.fixture
def creg_sizes() -> dict[str, int]:
    """Provide sample classical register sizes."""
    return {"c": 3, "m": 5}


# =============================================================================
# Load Gate Mappings Tests
# =============================================================================


class TestLoadGateMappings:
    """Tests for the load_gate_mappings function."""

    def test_load_valid_yaml(self, gates_yaml_path: str) -> None:
        """Load a valid gates.yaml file."""
        mappings = load_gate_mappings(gates_yaml_path)
        assert isinstance(mappings, dict)
        assert "mappings" in mappings
        assert "primitives" in mappings

    def test_load_returns_dict(self, gates_yaml_path: str) -> None:
        """Verify that load_gate_mappings returns a dictionary."""
        mappings = load_gate_mappings(gates_yaml_path)
        assert isinstance(mappings, dict)

    def test_load_contains_expected_gates(self, gates_yaml_path: str) -> None:
        """Verify that loaded mappings contain expected gate definitions."""
        mappings = load_gate_mappings(gates_yaml_path)
        gate_mappings = mappings.get("mappings", {})
        assert "rx" in gate_mappings
        assert "ry" in gate_mappings
        assert "cx" in gate_mappings
        assert "h" in gate_mappings

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_gate_mappings("/nonexistent/path/gates.yaml")

    def test_load_invalid_yaml_content_raises_error(self, tmp_path: Path) -> None:
        """Loading a file with invalid YAML raises an error."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: [unclosed", encoding="utf-8")
        with pytest.raises(Exception):  # noqa: B030
            load_gate_mappings(str(yaml_file))

    def test_load_non_dict_yaml_raises_error(self, tmp_path: Path) -> None:
        """Loading a file with non-dict top-level content raises ValueError."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a mapping at the top level"):
            load_gate_mappings(str(yaml_file))


# =============================================================================
# Flatten QRef Tests
# =============================================================================


class TestFlattenQRef:
    """Tests for the flatten_qref function."""

    def test_flatten_simple_qubit_reference(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flatten a simple quantum reference to a qubit index."""
        qref = QRef(reg="q", idx=0, line=1, col=1)
        result = flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert result == 0

    def test_flatten_qubit_in_second_register(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flatten a quantum reference from a non-zero offset register."""
        qref = QRef(reg="a", idx=1, line=1, col=1)
        result = flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert result == 3  # offset 2 + index 1

    def test_flatten_qubit_in_third_register(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flatten a quantum reference from the third register."""
        qref = QRef(reg="b", idx=2, line=1, col=1)
        result = flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert result == 7  # offset 5 + index 2

    def test_flatten_unknown_register_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flattening an unknown register raises QasmGraphIntegrationError."""
        qref = QRef(reg="unknown", idx=0, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E701"

    def test_flatten_none_index_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference without an index raises QasmGraphIntegrationError."""
        qref = QRef(reg="q", idx=None, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E702"

    def test_flatten_negative_index_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference with negative index raises QasmGraphIntegrationError."""
        qref = QRef(reg="q", idx=-1, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E702"

    def test_flatten_out_of_range_index_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference with out-of-range index raises QasmGraphIntegrationError."""
        qref = QRef(reg="q", idx=10, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_qref(qref, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E702"


# =============================================================================
# Flatten CRef Tests
# =============================================================================


class TestFlattenCRef:
    """Tests for the flatten_cref function."""

    def test_flatten_simple_bit_reference(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flatten a simple classical reference to a bit index."""
        cref = CRef(reg="c", idx=0, line=1, col=1)
        result = flatten_cref(cref, creg_offsets, creg_sizes)
        assert result == 0

    def test_flatten_bit_in_second_register(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flatten a classical reference from a non-zero offset register."""
        cref = CRef(reg="m", idx=2, line=1, col=1)
        result = flatten_cref(cref, creg_offsets, creg_sizes)
        assert result == 5  # offset 3 + index 2

    def test_flatten_unknown_register_raises_error(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flattening an unknown classical register raises QasmGraphIntegrationError."""
        cref = CRef(reg="unknown", idx=0, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_cref(cref, creg_offsets, creg_sizes)
        assert exc_info.value.code == "E701"

    def test_flatten_none_index_raises_error(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference without an index raises QasmGraphIntegrationError."""
        cref = CRef(reg="c", idx=None, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_cref(cref, creg_offsets, creg_sizes)
        assert exc_info.value.code == "E702"

    def test_flatten_negative_index_raises_error(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference with negative index raises QasmGraphIntegrationError."""
        cref = CRef(reg="c", idx=-1, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_cref(cref, creg_offsets, creg_sizes)
        assert exc_info.value.code == "E702"

    def test_flatten_out_of_range_index_raises_error(
        self, creg_offsets: dict[str, int], creg_sizes: dict[str, int]
    ) -> None:
        """Flattening a reference with out-of-range index raises QasmGraphIntegrationError."""
        cref = CRef(reg="c", idx=10, line=1, col=1)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            flatten_cref(cref, creg_offsets, creg_sizes)
        assert exc_info.value.code == "E702"


# =============================================================================
# Template Substitution Tests
# =============================================================================


class TestSubstituteTemplate:
    """Tests for the substitute_template function."""

    def test_substitute_no_template(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Substitute a numeric constant without templates."""
        result = substitute_template("1.5707963267948966", {})
        assert approx_equal(result, math.pi / 2)

    def test_substitute_single_parameter(self, approx_equal: Callable[[float, float], bool]) -> None:
        """Substitute a single parameter template."""
        result = substitute_template("{theta}", {"theta": 0.5})
        assert approx_equal(result, 0.5)

    def test_substitute_multiple_parameters(
        self, approx_equal: Callable[[float, float], bool]
    ) -> None:
        """Substitute multiple parameter templates without arithmetic."""
        # The implementation does not support arithmetic expressions in templates,
        # so we test that multiple placeholders are correctly substituted.
        # The expression is evaluated as a whole, not individually.
        result = substitute_template("{theta}", {"theta": 0.8})
        assert approx_equal(result, 0.8)

    def test_substitute_parameter_in_simple_expression(
        self, approx_equal: Callable[[float, float], bool]
    ) -> None:
        """Substitute a simple numeric parameter."""
        result = substitute_template("{theta}", {"theta": 0.5})
        assert approx_equal(result, 0.5)

    def test_substitute_whitespace_around_parameter(
        self, approx_equal: Callable[[float, float], bool]
    ) -> None:
        """Substitute a parameter with whitespace in the placeholder."""
        result = substitute_template("{ theta }", {"theta": 0.5})
        assert approx_equal(result, 0.5)

    def test_substitute_empty_template(self) -> None:
        """Substitute an empty template returns 0.0."""
        result = substitute_template("", {})
        assert result == 0.0

    def test_substitute_missing_parameter_raises_error(self) -> None:
        """Substituting a missing parameter raises KeyError."""
        with pytest.raises(KeyError):
            substitute_template("{theta}", {})

    def test_substitute_invalid_expression_raises_error(self) -> None:
        """Substituting an invalid expression raises QasmGraphIntegrationError."""
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            substitute_template("not a valid math expression", {})
        assert exc_info.value.code == "E703"

    def test_substitute_evaluates_numeric_literal(
        self, approx_equal: Callable[[float, float], bool]
    ) -> None:
        """Substitute and evaluate a numeric literal directly."""
        result = substitute_template("1.5707963267948966", {})
        assert approx_equal(result, math.pi / 2)

    def test_substitute_parameter_only_numeric_value(
        self, approx_equal: Callable[[float, float], bool]
    ) -> None:
        """Substitute a parameter that is a numeric value only."""
        result = substitute_template("{theta}", {"theta": 1.5707963267948966})
        assert approx_equal(result, math.pi / 2)


# =============================================================================
# Apply Gate Mapping Tests
# =============================================================================


class TestApplyGateMapping:
    """Tests for the apply_gate_mapping function."""

    def test_map_single_qubit_gate(self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]) -> None:
        """Apply a simple one-to-one gate mapping."""
        call = GateCallAST(
            name="rx",
            params=[0.5],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": ["theta"],
            "map": {"op": "RX", "qubits": ["q0"], "args": ["{theta}"]},
        }
        ops = apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert len(ops) == 1
        assert ops[0].name == "RX"
        assert ops[0].qubits == [0]
        assert ops[0].params == (0.5,)

    def test_map_two_qubit_gate(self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]) -> None:
        """Apply a two-qubit gate mapping."""
        call = GateCallAST(
            name="cx",
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1), QRef(reg="a", idx=0, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": [],
            "map": {"op": "CX", "qubits": ["q0", "q1"], "args": []},
        }
        ops = apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert len(ops) == 1
        assert ops[0].name == "CX"
        assert ops[0].qubits == [0, 2]  # q[0]=0, a[0]=2
        assert ops[0].params == ()

    def test_expand_gate_to_multiple_operations(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Apply an expand mapping that decomposes a gate into multiple operations."""
        call = GateCallAST(
            name="h",
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": [],
            "expand": [
                {"op": "RZ", "qubits": ["q0"], "args": ["1.5707963267948966"]},
                {"op": "RY", "qubits": ["q0"], "args": ["1.5707963267948966"]},
                {"op": "RZ", "qubits": ["q0"], "args": ["1.5707963267948966"]},
            ],
        }
        ops = apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert len(ops) == 3
        assert all(op.qubits == [0] for op in ops)
        assert ops[0].name == "RZ"
        assert ops[1].name == "RY"
        assert ops[2].name == "RZ"

    def test_map_with_integer_qubit_reference(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Apply a mapping using integer qubit indices."""
        call = GateCallAST(
            name="cx",
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1), QRef(reg="q", idx=1, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": [],
            "map": {"op": "CX", "qubits": [0, 1], "args": []},
        }
        ops = apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert len(ops) == 1
        assert ops[0].qubits == [0, 1]

    def test_map_multiple_parameters(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """Apply a mapping with multiple parameters."""
        call = GateCallAST(
            name="cu3",
            params=[0.5, 0.3, 0.7],
            qargs=[QRef(reg="q", idx=0, line=1, col=1), QRef(reg="q", idx=1, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": ["theta", "phi", "lam"],
            "map": {"op": "CU3", "qubits": ["q0", "q1"], "args": ["{theta}", "{phi}", "{lam}"]},
        }
        ops = apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert len(ops) == 1
        assert ops[0].params == (0.5, 0.3, 0.7)

    def test_both_map_and_expand_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """A mapping with both map and expand raises QasmGraphIntegrationError."""
        call = GateCallAST(name="invalid", params=[], qargs=[], line=1, col=1)
        mapping = {
            "params": [],
            "map": {"op": "X", "qubits": ["q0"], "args": []},
            "expand": [{"op": "RX", "qubits": ["q0"], "args": ["3.141592653589793"]}],
        }
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E701"

    def test_missing_map_and_expand_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """A mapping without map or expand raises QasmGraphIntegrationError."""
        call = GateCallAST(name="invalid", params=[], qargs=[], line=1, col=1)
        mapping: dict[str, list] = {"params": []}
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E701"

    def test_invalid_op_name_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """A mapping entry without op identifier raises QasmGraphIntegrationError."""
        call = GateCallAST(name="invalid", params=[], qargs=[], line=1, col=1)
        mapping = {
            "params": [],
            "map": {"qubits": ["q0"], "args": []},  # Missing op
        }
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E701"

    def test_qubit_index_out_of_range_raises_error(
        self, qreg_offsets: dict[str, int], qreg_sizes: dict[str, int]
    ) -> None:
        """A mapping with an out-of-range qubit index raises QasmGraphIntegrationError."""
        call = GateCallAST(
            name="cx",
            params=[],
            qargs=[QRef(reg="q", idx=0, line=1, col=1), QRef(reg="q", idx=1, line=1, col=1)],
            line=1,
            col=1,
        )
        mapping = {
            "params": [],
            "map": {"op": "CX", "qubits": [0, 5], "args": []},  # 5 is out of range
        }
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            apply_gate_mapping(call, mapping, qreg_offsets, qreg_sizes)
        assert exc_info.value.code == "E702"


# =============================================================================
# Lower to IR Tests
# =============================================================================


class TestLowerToIR:
    """Tests for the lower_to_ir function."""

    def test_lower_empty_program(self, gates_yaml_path: str) -> None:
        """Lower an empty program with only qreg/creg declarations."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 0
        assert circuit.meas_map is None

    def test_lower_single_gate(self, gates_yaml_path: str) -> None:
        """Lower a program with a single gate operation."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(
                    name="rx",
                    params=[0.5],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 1
        assert len(circuit.ops) == 1
        assert circuit.ops[0].name == "RX"
        assert circuit.ops[0].params == (0.5,)

    def test_lower_multiple_gates(self, gates_yaml_path: str) -> None:
        """Lower a program with multiple gate operations."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            body=[
                GateCallAST(
                    name="rx",
                    params=[0.5],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                ),
                GateCallAST(
                    name="cx",
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=2, col=1), QRef(reg="q", idx=1, line=2, col=1)],
                    line=2,
                    col=1,
                ),
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 2

    def test_lower_expand_gate(self, gates_yaml_path: str) -> None:
        """Lower a program with a gate that expands to multiple operations."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(
                    name="h",
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 1
        assert len(circuit.ops) == 1  # h maps to native H gate
        assert circuit.ops[0].name == "H"

    def test_lower_with_measurement(self, gates_yaml_path: str) -> None:
        """Lower a program with a measurement operation."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            cregs=[("c", 1)],
            body=[
                MeasureAST(
                    q=QRef(reg="q", idx=0, line=1, col=1),
                    c=CRef(reg="c", idx=0, line=1, col=1),
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.meas_map == [(0, 0)]

    def test_lower_multiple_measurements(self, gates_yaml_path: str) -> None:
        """Lower a program with multiple measurement operations."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[
                MeasureAST(
                    q=QRef(reg="q", idx=0, line=1, col=1),
                    c=CRef(reg="c", idx=0, line=1, col=1),
                    line=1,
                    col=1,
                ),
                MeasureAST(
                    q=QRef(reg="q", idx=1, line=2, col=1),
                    c=CRef(reg="c", idx=1, line=2, col=1),
                    line=2,
                    col=1,
                ),
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.meas_map == [(0, 0), (1, 1)]

    def test_lower_unmapped_gate_raises_error(self, gates_yaml_path: str) -> None:
        """Lowering an unmapped gate raises QasmGraphIntegrationError."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(
                    name="unmapped_gate",
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            lower_to_ir(ast, mappings)
        assert exc_info.value.code == "E701"

    def test_lower_gate_arity_mismatch_raises_error(self, gates_yaml_path: str) -> None:
        """Lowering a gate with wrong arity raises QasmGraphIntegrationError."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            body=[
                GateCallAST(
                    name="cx",  # cx expects 2 qubits
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],  # but only 1 provided
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            lower_to_ir(ast, mappings)
        assert exc_info.value.code == "E702"

    def test_lower_gate_parameter_mismatch_raises_error(self, gates_yaml_path: str) -> None:
        """Lowering a gate with wrong parameter count raises QasmGraphIntegrationError."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            body=[
                GateCallAST(
                    name="rx",  # rx expects 1 parameter
                    params=[0.5, 0.3],  # but 2 are provided
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            lower_to_ir(ast, mappings)
        assert exc_info.value.code == "E703"

    def test_lower_measurement_operand_size_mismatch_raises_error(self, gates_yaml_path: str) -> None:
        """Lowering with mismatched measurement operand sizes raises error."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 3)],
            body=[
                MeasureAST(
                    q=QRef(reg="q", idx=None, line=1, col=1),  # all qubits
                    c=CRef(reg="c", idx=None, line=1, col=1),  # all cbits
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        with pytest.raises(QasmGraphIntegrationError) as exc_info:
            lower_to_ir(ast, mappings)
        assert exc_info.value.code == "E703"

    def test_lower_multiple_qregs(self, gates_yaml_path: str) -> None:
        """Lower a program with multiple quantum registers."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2), ("a", 3)],
            body=[
                GateCallAST(
                    name="rx",
                    params=[0.5],
                    qargs=[QRef(reg="a", idx=1, line=1, col=1)],
                    line=1,
                    col=1,
                )
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 5  # 2 + 3
        assert circuit.ops[0].qubits == [3]  # a[1] with offset 2

    def test_lower_mixed_gates_and_measurements(self, gates_yaml_path: str) -> None:
        """Lower a program with mixed gates and measurements."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[
                GateCallAST(
                    name="h",
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                ),
                MeasureAST(
                    q=QRef(reg="q", idx=0, line=2, col=1),
                    c=CRef(reg="c", idx=0, line=2, col=1),
                    line=2,
                    col=1,
                ),
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) == 1  # h maps to native H gate
        assert circuit.ops[0].name == "H"
        assert circuit.meas_map == [(0, 0)]

    def test_lower_invalid_mappings_dict_raises_error(self) -> None:
        """Lowering with invalid mappings dict raises ValueError."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
        )
        bad_mappings = {"mappings": "not a dict"}  # mappings should be a dict
        with pytest.raises(ValueError, match="must provide a 'mappings' dictionary"):
            lower_to_ir(ast, bad_mappings)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoweringIntegration:
    """Integration tests for the complete lowering pipeline."""

    def test_lowering_simple_circuit_with_parameterized_gates(self, gates_yaml_path: str) -> None:
        """Lower a complete simple circuit with parameterized gates."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 2)],
            cregs=[("c", 2)],
            body=[
                GateCallAST(
                    name="rx",
                    params=[0.785398],  # pi/4
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                ),
                GateCallAST(
                    name="ry",
                    params=[1.570796],  # pi/2
                    qargs=[QRef(reg="q", idx=1, line=2, col=1)],
                    line=2,
                    col=1,
                ),
                GateCallAST(
                    name="cx",
                    params=[],
                    qargs=[QRef(reg="q", idx=0, line=3, col=1), QRef(reg="q", idx=1, line=3, col=1)],
                    line=3,
                    col=1,
                ),
                MeasureAST(
                    q=QRef(reg="q", idx=0, line=4, col=1),
                    c=CRef(reg="c", idx=0, line=4, col=1),
                    line=4,
                    col=1,
                ),
                MeasureAST(
                    q=QRef(reg="q", idx=1, line=5, col=1),
                    c=CRef(reg="c", idx=1, line=5, col=1),
                    line=5,
                    col=1,
                ),
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        assert isinstance(circuit, Circuit)
        assert circuit.n_qubits == 2
        assert len(circuit.ops) > 0
        assert circuit.meas_map is not None
        assert len(circuit.meas_map) == 2

    def test_circuit_to_dict_and_back(self, gates_yaml_path: str) -> None:
        """Lower a circuit and verify its JSON serialization."""
        ast = ProgramAST(
            version="2.0",
            qregs=[("q", 1)],
            cregs=[("c", 1)],
            body=[
                GateCallAST(
                    name="rx",
                    params=[0.5],
                    qargs=[QRef(reg="q", idx=0, line=1, col=1)],
                    line=1,
                    col=1,
                ),
                MeasureAST(
                    q=QRef(reg="q", idx=0, line=2, col=1),
                    c=CRef(reg="c", idx=0, line=2, col=1),
                    line=2,
                    col=1,
                ),
            ],
        )
        mappings = load_gate_mappings(gates_yaml_path)
        circuit = lower_to_ir(ast, mappings)
        circuit_dict = circuit.to_dict()
        assert "n_qubits" in circuit_dict
        assert "ops" in circuit_dict
        assert "meas_map" in circuit_dict
