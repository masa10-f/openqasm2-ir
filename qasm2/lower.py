"""Lowering utilities from the normalized OpenQASM 2 AST to the custom IR."""

from __future__ import annotations

import importlib.resources as importlib_resources
import re
from pathlib import Path
from typing import Dict, List

import yaml

from ir.circuit import Circuit, Op
from qasm2.ast_nodes import CRef, GateCallAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmGraphIntegrationError
from qasm2.parser import parse_qasm

__all__ = [
    "load_default_gate_mappings",
    "load_gate_mappings",
    "lower_to_ir",
    "flatten_qref",
    "flatten_cref",
    "substitute_template",
    "apply_gate_mapping",
]

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


def load_default_gate_mappings() -> dict:
    """Load the default gate mappings from the packaged gates.yaml resource.

    This is a convenience function that locates and loads the default gate
    mapping configuration without requiring the user to specify a path.

    Returns
    -------
    dict
        Parsed YAML content describing the gate set.

    Raises
    ------
    FileNotFoundError
        If the gates.yaml resource cannot be located.
    ValueError
        If the file contents cannot be parsed into a dictionary.

    Examples
    --------
    >>> from qasm2.lower import load_default_gate_mappings, lower_to_ir
    >>> from qasm2.normalize import normalize_program
    >>> gate_mappings = load_default_gate_mappings()
    >>> circuit = lower_to_ir(normalized_ast, gate_mappings)
    """
    # Try to locate and read gates.yaml as a packaged resource
    candidates = ("gates", "qasm2.gates", "qasm2")
    for package in candidates:
        try:
            root = importlib_resources.files(package)
        except (ModuleNotFoundError, AttributeError, TypeError):
            continue
        candidate = root.joinpath("gates.yaml")
        try:
            if candidate.is_file():
                content = candidate.read_text(encoding="utf-8")
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    raise ValueError(f"Gate mapping must contain a mapping at the top level.")
                return data
        except (AttributeError, FileNotFoundError):
            continue

    # Fallback to development tree location
    local_fallback = Path(__file__).resolve().parent.parent / "gates" / "gates.yaml"
    if local_fallback.exists():
        content = local_fallback.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError(f"Gate mapping must contain a mapping at the top level.")
        return data

    raise FileNotFoundError(
        "Unable to locate packaged gates.yaml. "
        "Please ensure the package is properly installed or provide an explicit path to load_gate_mappings()."
    )


def load_gate_mappings(yaml_path: str) -> dict:
    """Load gate lowering definitions from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the ``gates.yaml`` file that defines primitive mappings.

    Returns
    -------
    dict
        Parsed YAML content describing the gate set.

    Raises
    ------
    FileNotFoundError
        If the supplied path does not exist.
    ValueError
        If the file contents cannot be parsed into a dictionary.
    """
    path = Path(yaml_path)
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError(f"Gate mapping file '{yaml_path}' must contain a mapping at the top level.")
    return data


def lower_to_ir(ast: ProgramAST, gate_mappings: dict) -> Circuit:
    """Convert a normalized program AST to the intermediate circuit IR.

    Parameters
    ----------
    ast : ProgramAST
        Normalized OpenQASM 2 abstract syntax tree.
    gate_mappings : dict
        Gate lowering rules loaded from :func:`load_gate_mappings`.

    Returns
    -------
    Circuit
        Lowered intermediate representation circuit.

    Raises
    ------
    QasmGraphIntegrationError
        If the lowering encounters an unmapped gate or inconsistent signature.
    """
    qreg_sizes: Dict[str, int] = {name: size for name, size in ast.qregs}
    qreg_offsets: Dict[str, int] = {}
    q_offset = 0
    for name, size in ast.qregs:
        qreg_offsets[name] = q_offset
        q_offset += size

    creg_sizes: Dict[str, int] = {name: size for name, size in ast.cregs}
    creg_offsets: Dict[str, int] = {}
    c_offset = 0
    for name, size in ast.cregs:
        creg_offsets[name] = c_offset
        c_offset += size

    mappings = gate_mappings.get("mappings", {})
    if not isinstance(mappings, dict):
        raise ValueError("Gate mappings must provide a 'mappings' dictionary.")

    ops: List[Op] = []
    meas_map: List[tuple[int, int]] = []

    for stmt in ast.body:
        if isinstance(stmt, GateCallAST):
            mapping = mappings.get(stmt.name)
            if mapping is None:
                raise QasmGraphIntegrationError(
                    "E701",
                    f"gate '{stmt.name}' has no mapping",
                    stmt.line,
                    stmt.col,
                )
            expected_params = list(mapping.get("params", []))
            if len(stmt.params) != len(expected_params):
                raise QasmGraphIntegrationError(
                    "E703",
                    f"gate '{stmt.name}' expects {len(expected_params)} parameters but received {len(stmt.params)}",
                    stmt.line,
                    stmt.col,
                )
            expected_arity = _infer_expected_arity(mapping)
            if expected_arity is not None and len(stmt.qargs) != expected_arity:
                raise QasmGraphIntegrationError(
                    "E702",
                    f"gate '{stmt.name}' expects {expected_arity} qubits but received {len(stmt.qargs)}",
                    stmt.line,
                    stmt.col,
                )
            ops.extend(apply_gate_mapping(stmt, mapping, qreg_offsets, qreg_sizes))
            continue

        if isinstance(stmt, MeasureAST):
            q_indices = _expand_qref(stmt.q, qreg_offsets, qreg_sizes)
            c_indices = _expand_cref(stmt.c, creg_offsets, creg_sizes)
            if len(q_indices) != len(c_indices):
                raise QasmGraphIntegrationError(
                    "E703",
                    "measurement operands do not match in size",
                    stmt.line,
                    stmt.col,
                )
            meas_map.extend(zip(q_indices, c_indices))
            continue

    return Circuit(
        n_qubits=q_offset,
        ops=ops,
        meas_map=meas_map or None,
    )


def flatten_qref(qref: QRef, qreg_offsets: Dict[str, int], qreg_sizes: Dict[str, int]) -> int:
    """Convert a quantum reference to a flat qubit index.

    Parameters
    ----------
    qref : QRef
        Quantum reference from the AST.
    qreg_offsets : Dict[str, int]
        Mapping of quantum register names to their cumulative offsets.
    qreg_sizes : Dict[str, int]
        Mapping of register names to their declared sizes.

    Returns
    -------
    int
        Flat qubit index suitable for the IR.

    Raises
    ------
    QasmGraphIntegrationError
        If the reference does not resolve to a concrete qubit.
    """
    offset = qreg_offsets.get(qref.reg)
    size = qreg_sizes.get(qref.reg)
    if offset is None or size is None:
        raise QasmGraphIntegrationError("E701", f"unknown quantum register '{qref.reg}'", qref.line, qref.col)
    if qref.idx is None:
        raise QasmGraphIntegrationError(
            "E702",
            f"quantum reference '{qref.reg}' must address a specific qubit during lowering",
            qref.line,
            qref.col,
        )
    if qref.idx < 0 or qref.idx >= size:
        raise QasmGraphIntegrationError(
            "E702",
            f"qubit index {qref.idx} out of range for register '{qref.reg}'",
            qref.line,
            qref.col,
        )
    return offset + qref.idx


def flatten_cref(cref: CRef, creg_offsets: Dict[str, int], creg_sizes: Dict[str, int]) -> int:
    """Convert a classical reference to a flat bit index.

    Parameters
    ----------
    cref : CRef
        Classical reference from the AST.
    creg_offsets : Dict[str, int]
        Mapping of classical register names to their cumulative offsets.
    creg_sizes : Dict[str, int]
        Mapping of classical register names to their declared sizes.

    Returns
    -------
    int
        Flat classical bit index.

    Raises
    ------
    QasmGraphIntegrationError
        If the reference does not resolve to a concrete classical bit.
    """
    offset = creg_offsets.get(cref.reg)
    size = creg_sizes.get(cref.reg)
    if offset is None or size is None:
        raise QasmGraphIntegrationError("E701", f"unknown classical register '{cref.reg}'", cref.line, cref.col)
    if cref.idx is None:
        raise QasmGraphIntegrationError(
            "E702",
            f"classical reference '{cref.reg}' must address a specific bit during lowering",
            cref.line,
            cref.col,
        )
    if cref.idx < 0 or cref.idx >= size:
        raise QasmGraphIntegrationError(
            "E702",
            f"classical index {cref.idx} out of range for register '{cref.reg}'",
            cref.line,
            cref.col,
        )
    return offset + cref.idx


def substitute_template(template: str, param_map: Dict[str, float]) -> float:
    """Evaluate a gate argument template using the supplied parameter values.

    Parameters
    ----------
    template : str
        Expression template from the mapping definition.
    param_map : Dict[str, float]
        Values of the gate parameters keyed by their placeholder names.

    Returns
    -------
    float
        Evaluated numeric argument expressed in radians.

    Raises
    ------
    KeyError
        If the template references an unknown parameter placeholder.
    QasmGraphIntegrationError
        If expression evaluation fails.
    """
    if not isinstance(template, str):
        return float(template)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in param_map:
            raise KeyError(key)
        return repr(float(param_map[key]))

    expression = _PLACEHOLDER_PATTERN.sub(_replace, template).strip()
    if not expression:
        return 0.0

    try:
        return float(expression)
    except ValueError:
        try:
            return _evaluate_expression(expression)
        except QasmGraphIntegrationError:
            raise
        except Exception as exc:
            raise QasmGraphIntegrationError("E703", f"failed to evaluate template '{template}'", 1, 1) from exc


def apply_gate_mapping(
    call: GateCallAST,
    mapping: dict,
    qreg_offsets: Dict[str, int],
    qreg_sizes: Dict[str, int],
) -> List[Op]:
    """Apply a gate mapping entry to produce IR operations.

    Parameters
    ----------
    call : GateCallAST
        Gate invocation being lowered.
    mapping : dict
        Mapping definition describing how to translate the gate.
    qreg_offsets : Dict[str, int]
        Quantum register offsets.
    qreg_sizes : Dict[str, int]
        Quantum register sizes.

    Returns
    -------
    List[Op]
        Sequence of IR operations implementing the gate call.

    Raises
    ------
    QasmGraphIntegrationError
        If the mapping definition is malformed or references invalid operands.
    """
    param_list = list(mapping.get("params", []))
    param_map = {name: float(value) for name, value in zip(param_list, call.params)}

    qubit_indices = [flatten_qref(qref, qreg_offsets, qreg_sizes) for qref in call.qargs]

    if "map" in mapping and "expand" in mapping:
        raise QasmGraphIntegrationError(
            "E701",
            f"gate '{call.name}' mapping cannot define both 'map' and 'expand'",
            call.line,
            call.col,
        )

    entries: List[dict] = []
    if "map" in mapping:
        entry = mapping["map"]
        if not isinstance(entry, dict):
            raise QasmGraphIntegrationError("E701", f"gate '{call.name}' map entry is invalid", call.line, call.col)
        entries = [entry]
    elif "expand" in mapping:
        raw_expand = mapping["expand"]
        if not isinstance(raw_expand, list):
            raise QasmGraphIntegrationError(
                "E701",
                f"gate '{call.name}' expand entry must be a list",
                call.line,
                call.col,
            )
        entries = raw_expand
    else:
        raise QasmGraphIntegrationError(
            "E701",
            f"gate '{call.name}' mapping requires 'map' or 'expand'",
            call.line,
            call.col,
        )

    resolved_ops: List[Op] = []
    for entry in entries:
        op_name = entry.get("op")
        if not isinstance(op_name, str):
            raise QasmGraphIntegrationError(
                "E701",
                f"gate '{call.name}' mapping entry is missing an 'op' identifier",
                call.line,
                call.col,
            )
        raw_qubits = entry.get("qubits", [])
        if not isinstance(raw_qubits, list):
            raise QasmGraphIntegrationError(
                "E701",
                f"gate '{call.name}' mapping entry must provide qubit identifiers as a list",
                call.line,
                call.col,
            )
        qubits = [_resolve_qubit_reference(label, qubit_indices, call) for label in raw_qubits]
        raw_args = entry.get("args", [])
        if not isinstance(raw_args, list):
            raise QasmGraphIntegrationError(
                "E701",
                f"gate '{call.name}' mapping entry must provide args as a list",
                call.line,
                call.col,
            )
        params: List[float] = []
        for template in raw_args:
            try:
                params.append(substitute_template(template, param_map))
            except KeyError as exc:
                missing = exc.args[0]
                raise QasmGraphIntegrationError(
                    "E703",
                    f"gate '{call.name}' template references unknown parameter '{missing}'",
                    call.line,
                    call.col,
                ) from exc
            except QasmGraphIntegrationError as exc:
                raise QasmGraphIntegrationError(
                    exc.code,
                    exc.message,
                    call.line,
                    call.col,
                ) from exc
        resolved_ops.append(Op(name=op_name, qubits=qubits, params=tuple(params)))
    return resolved_ops


def _expand_qref(qref: QRef, qreg_offsets: Dict[str, int], qreg_sizes: Dict[str, int]) -> List[int]:
    if qref.idx is not None:
        return [flatten_qref(qref, qreg_offsets, qreg_sizes)]
    size = qreg_sizes.get(qref.reg)
    offset = qreg_offsets.get(qref.reg)
    if size is None or offset is None:
        raise QasmGraphIntegrationError("E701", f"unknown quantum register '{qref.reg}'", qref.line, qref.col)
    return [offset + idx for idx in range(size)]


def _expand_cref(cref: CRef, creg_offsets: Dict[str, int], creg_sizes: Dict[str, int]) -> List[int]:
    if cref.idx is not None:
        return [flatten_cref(cref, creg_offsets, creg_sizes)]
    size = creg_sizes.get(cref.reg)
    offset = creg_offsets.get(cref.reg)
    if size is None or offset is None:
        raise QasmGraphIntegrationError("E701", f"unknown classical register '{cref.reg}'", cref.line, cref.col)
    return [offset + idx for idx in range(size)]


def _resolve_qubit_reference(label: object, qubit_indices: List[int], call: GateCallAST) -> int:
    if isinstance(label, int):
        if label < 0 or label >= len(qubit_indices):
            raise QasmGraphIntegrationError(
                "E702",
                f"gate '{call.name}' mapping references qubit position {label} out of range",
                call.line,
                call.col,
            )
        return qubit_indices[label]
    if not isinstance(label, str):
        raise QasmGraphIntegrationError(
            "E702",
            f"gate '{call.name}' mapping references invalid qubit identifier '{label}'",
            call.line,
            call.col,
        )
    label = label.strip()
    match = re.fullmatch(r"q(\d+)", label)
    if match:
        index = int(match.group(1))
        if index < 0 or index >= len(qubit_indices):
            raise QasmGraphIntegrationError(
                "E702",
                f"gate '{call.name}' mapping references qubit position {index} out of range",
                call.line,
                call.col,
            )
        return qubit_indices[index]
    raise QasmGraphIntegrationError(
        "E702",
        f"gate '{call.name}' mapping references unsupported qubit label '{label}'",
        call.line,
        call.col,
    )


def _evaluate_expression(expression: str) -> float:
    snippet = "OPENQASM 2.0;\nqreg q[1];\nrx(" + expression + ") q[0];\n"
    program = parse_qasm(snippet)
    if not program.body:
        raise QasmGraphIntegrationError("E703", f"failed to evaluate expression '{expression}'", 1, 1)
    stmt = program.body[0]
    if not isinstance(stmt, GateCallAST) or not stmt.params:
        raise QasmGraphIntegrationError("E703", f"failed to evaluate expression '{expression}'", 1, 1)
    return float(stmt.params[0])


def _infer_expected_arity(mapping: dict) -> int | None:
    if "arity" in mapping:
        try:
            return int(mapping["arity"])
        except (TypeError, ValueError):
            return None
    if "map" in mapping and isinstance(mapping["map"], dict):
        entries = [mapping["map"]]
    elif "expand" in mapping and isinstance(mapping["expand"], list):
        entries = list(mapping["expand"])
    else:
        entries = []
    max_index = -1
    for entry in entries:
        qubits = entry.get("qubits")
        if not isinstance(qubits, list):
            continue
        for label in qubits:
            if isinstance(label, int):
                max_index = max(max_index, label)
                continue
            if isinstance(label, str):
                match = re.fullmatch(r"q(\d+)", label.strip())
                if match:
                    max_index = max(max_index, int(match.group(1)))
    if max_index >= 0:
        return max_index + 1
    return None
