from __future__ import annotations

import ast
import importlib.resources as importlib_resources
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from lark import Lark, Tree, Token
from lark.exceptions import LarkError, UnexpectedInput
from lark.visitors import Transformer

from qasm2 import expr_eval
from qasm2.ast_nodes import BarrierAST, CRef, GateCallAST, GateDefAST, MeasureAST, ProgramAST, QRef
from qasm2.errors import QasmError

__all__ = ["QasmTransformer", "create_parser", "parse_qasm", "parse_qasm_file"]

_GRAMMAR_PACKAGE = "qasm2.grammar"
_GRAMMAR_FILE = "qasm2_strict.lark"


@dataclass(frozen=True)
class _GateContext:
    """Book-keeping for the gate definition analysis phase."""

    name: str
    params: frozenset[str]
    qargs: frozenset[str]


class QasmTransformer(Transformer):
    """Convert a Lark parse tree into the Project AST representation."""

    _BUILTIN_GATES: frozenset[str] = frozenset(
        {
            "u1",
            "u2",
            "u3",
            "rx",
            "ry",
            "rz",
            "p",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "id",
            "sx",
            "sxdg",
            "swap",
            "cx",
            "cz",
            "ccx",
            "cswap",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._reset_state()

    def transform(self, tree: Tree) -> ProgramAST:
        """Perform the tree-to-AST conversion."""
        if not isinstance(tree, Tree) or tree.data != "start":
            raise ValueError("Transformation requires the `start` rule as entry point.")
        self._reset_state()
        self._visit_start(tree)
        return ProgramAST(
            version=self._version,
            includes=list(self._includes),
            qregs=list(self._qregs),
            cregs=list(self._cregs),
            gate_defs=dict(self._gate_defs),
            body=list(self._body),
        )

    def _reset_state(self) -> None:
        self._version: str = "2.0"
        self._includes: List[str] = []
        self._qregs: List[tuple[str, int]] = []
        self._cregs: List[tuple[str, int]] = []
        self._gate_defs: dict[str, GateDefAST] = {}
        self._body: List[GateCallAST | MeasureAST | BarrierAST] = []
        self._qreg_sizes: dict[str, int] = {}
        self._creg_sizes: dict[str, int] = {}
        self._gate_context: list[_GateContext] = []

    # -----------------------------------------------------------------
    # Top-level traversal

    def _visit_start(self, node: Tree) -> None:
        for child in node.children:
            if not isinstance(child, Tree):
                continue
            if child.data == "header":
                self._version = self._visit_header(child)
            elif child.data == "include":
                include = self._visit_include(child)
                if include is not None:
                    self._includes.append(include)
            elif child.data == "decl":
                self._visit_decl(child)
            elif child.data == "stmt":
                stmt = self._visit_stmt(child)
                if stmt is not None:
                    self._body.append(stmt)
            else:
                line, col = _node_location(child)
                raise QasmError("E401", f"Unexpected top-level construct '{child.data}'.", line, col)

    def _visit_header(self, node: Tree) -> str:
        number = _find_token(node.children, "NUMBER")
        if number is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed OPENQASM header.", line, col)
        return number.value

    def _visit_include(self, node: Tree) -> Optional[str]:
        token = _find_token(node.children, "STRING")
        if token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed include directive.", line, col)
        try:
            return ast.literal_eval(token.value)
        except (SyntaxError, ValueError) as exc:
            line, col = token.line, token.column
            raise QasmError("E401", f"Invalid include path {token.value!r}.", line, col) from exc

    # -----------------------------------------------------------------
    # Declarations

    def _visit_decl(self, node: Tree) -> None:
        if not node.children:
            line, col = _node_location(node)
            raise QasmError("E401", "Empty declaration encountered.", line, col)
        child = node.children[0]
        if not isinstance(child, Tree):
            line, col = _node_location(node)
            raise QasmError("E401", "Unexpected token in declaration.", line, col)
        if child.data == "qreg":
            self._visit_qreg(child)
        elif child.data == "creg":
            self._visit_creg(child)
        elif child.data == "gate_decl":
            self._visit_gate_decl(child)
        else:
            line, col = _node_location(child)
            raise QasmError("E401", f"Unsupported declaration '{child.data}'.", line, col)

    def _visit_qreg(self, node: Tree) -> None:
        name_token = _find_token(node.children, "CNAME")
        size_token = _find_token(node.children, "INT")
        if name_token is None or size_token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed qreg declaration.", line, col)
        name = name_token.value
        size = int(size_token.value)
        if name in self._qreg_sizes:
            line, col = name_token.line, name_token.column
            raise QasmError("E402", f"Duplicate quantum register '{name}'.", line, col)
        self._qreg_sizes[name] = size
        self._qregs.append((name, size))

    def _visit_creg(self, node: Tree) -> None:
        name_token = _find_token(node.children, "CNAME")
        size_token = _find_token(node.children, "INT")
        if name_token is None or size_token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed creg declaration.", line, col)
        name = name_token.value
        size = int(size_token.value)
        if name in self._creg_sizes:
            line, col = name_token.line, name_token.column
            raise QasmError("E402", f"Duplicate classical register '{name}'.", line, col)
        self._creg_sizes[name] = size
        self._cregs.append((name, size))

    def _visit_gate_decl(self, node: Tree) -> None:
        children = list(node.children)
        name_token = children[0] if children and isinstance(children[0], Token) else None
        if name_token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Gate declaration missing identifier.", line, col)
        gate_name = name_token.value
        idx = 1
        params: List[str] = []
        if idx < len(children):
            current_child = children[idx]
            if isinstance(current_child, Tree) and current_child.data == "gate_params":
                params = self._extract_identifier_list(current_child)
                idx += 1
        if idx >= len(children) or not isinstance(children[idx], Tree):
            line, col = _node_location(node)
            raise QasmError("E401", f"Gate '{gate_name}' missing argument list.", line, col)
        args_child = children[idx]
        assert isinstance(args_child, Tree)  # type check passed above
        if args_child.data != "gate_args":
            line, col = _node_location(node)
            raise QasmError("E401", f"Gate '{gate_name}' missing argument list.", line, col)
        gate_args = self._extract_identifier_list(args_child)
        idx += 1

        ctx = _GateContext(gate_name, frozenset(params), frozenset(gate_args))
        self._gate_context.append(ctx)
        body: List[GateCallAST] = []
        try:
            for body_node in children[idx:]:
                if not isinstance(body_node, Tree):
                    continue
                if body_node.data != "gate_body":
                    line, col = _node_location(body_node)
                    raise QasmError("E401", f"Unexpected construct in gate body: '{body_node.data}'.", line, col)
                gate_call = self._visit_gate_body(body_node)
                body.append(gate_call)
        finally:
            self._gate_context.pop()

        line, col = _node_location(node)
        gate_def = GateDefAST(name=gate_name, line=line, col=col, params=params, qargs=gate_args, body=body)
        if gate_name in self._gate_defs:
            raise QasmError("E402", f"Duplicate gate definition '{gate_name}'.", line, col)
        self._gate_defs[gate_name] = gate_def

    def _visit_gate_body(self, node: Tree) -> GateCallAST:
        if not node.children:
            line, col = _node_location(node)
            raise QasmError("E401", "Empty statement in gate body.", line, col)
        operand = node.children[0]
        if not isinstance(operand, Tree):
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed gate body statement.", line, col)
        if operand.data in {"gate_call", "uop"}:
            gate_call = self._parse_gate_call_like(operand)
            return gate_call
        if operand.data == "cx":
            return self._visit_cx(operand)
        if operand.data == "barrier":
            line, col = _node_location(operand)
            raise QasmError("E402", "Barrier not permitted inside gate bodies.", line, col)
        line, col = _node_location(operand)
        raise QasmError("E401", f"Unsupported gate body statement '{operand.data}'.", line, col)

    def _extract_identifier_list(self, node: Tree) -> List[str]:
        identifiers: List[str] = []
        for child in node.children:
            if isinstance(child, Token) and child.type == "CNAME":
                identifiers.append(child.value)
            elif isinstance(child, Tree):
                extracted = self._extract_identifier_list(child)
                identifiers.extend(extracted)
        return identifiers

    # -----------------------------------------------------------------
    # Statements

    def _visit_stmt(self, node: Tree) -> GateCallAST | MeasureAST | BarrierAST | None:
        for child in node.children:
            if not isinstance(child, Tree):
                continue
            if child.data in {"uop", "gate_call"}:
                return self._parse_gate_call_like(child)
            if child.data == "cx":
                return self._visit_cx(child)
            if child.data == "barrier":
                return self._visit_barrier(child)
            if child.data == "measure":
                return self._visit_measure(child)
            if child.data in {"reset_stmt", "if_stmt"}:
                line, col = _node_location(child)
                raise QasmError("E401", f"Unsupported statement '{child.data}'.", line, col)
            line, col = _node_location(child)
            raise QasmError("E401", f"Unsupported statement '{child.data}'.", line, col)
        return None

    def _parse_gate_call_like(self, node: Tree) -> GateCallAST:
        name_token = _find_first_identifier(node.children)
        if name_token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Gate invocation missing identifier.", line, col)
        gate_name = name_token.value
        params_tree = _find_tree(node.children, {"call_params", "gate_call_params"})
        qargs_nodes: List[Tree] = []
        if node.data == "gate_call":
            id_list = _find_tree(node.children, {"id_list"})
            if id_list is None:
                line, col = _node_location(node)
                raise QasmError("E401", f"Gate call '{gate_name}' missing argument list.", line, col)
            qargs_nodes = [child for child in id_list.children if isinstance(child, Tree)]
        else:
            qarg_tree = _find_tree(node.children, {"qarg"})
            if qarg_tree is None:
                line, col = _node_location(node)
                raise QasmError("E401", f"Gate call '{gate_name}' missing quantum operand.", line, col)
            qargs_nodes = [qarg_tree]

        params = self._evaluate_params(params_tree)
        qargs = [self._build_qref(qarg_node) for qarg_node in qargs_nodes]
        line, col = _node_location(node)
        self._ensure_gate_known(gate_name, line, col)
        return GateCallAST(name=gate_name, line=line, col=col, params=params, qargs=qargs)

    def _visit_cx(self, node: Tree) -> GateCallAST:
        qargs = [self._build_qref(child) for child in node.children if isinstance(child, Tree)]
        if len(qargs) != 2:
            line, col = _node_location(node)
            raise QasmError("E401", "Controlled-X requires two quantum operands.", line, col)
        line, col = _node_location(node)
        self._ensure_gate_known("cx", line, col)
        return GateCallAST(name="cx", line=line, col=col, params=[], qargs=qargs)

    def _visit_barrier(self, node: Tree) -> BarrierAST:
        qarg_list = _find_tree(node.children, {"qarg_list"})
        qargs = []
        if qarg_list is not None:
            qargs = [self._build_qref(child) for child in qarg_list.children if isinstance(child, Tree)]
        else:
            qargs = [self._build_qref(child) for child in node.children if isinstance(child, Tree)]
        line, col = _node_location(node)
        return BarrierAST(line=line, col=col, qargs=qargs)

    def _visit_measure(self, node: Tree) -> MeasureAST:
        qarg_tree = _find_tree(node.children, {"qarg"})
        carg_tree = _find_tree(node.children, {"carg"})
        if qarg_tree is None or carg_tree is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Malformed measurement statement.", line, col)
        qref = self._build_qref(qarg_tree)
        cref = self._build_cref(carg_tree)
        line, col = _node_location(node)
        return MeasureAST(q=qref, c=cref, line=line, col=col)

    # -----------------------------------------------------------------
    # Helpers

    def _evaluate_params(self, node: Optional[Tree]) -> List[float]:
        if node is None:
            return []
        values: List[float] = []
        for child in node.children:
            if isinstance(child, (Tree, Token)):
                # In gate definition context, preserve symbolic parameter names as strings
                # They will be substituted during normalization
                if self._gate_context and isinstance(child, Tree) and child.data == "param_ref":
                    if child.children and isinstance(child.children[0], Token):
                        param_name = child.children[0].value
                        values.append(param_name)  # normalize.py expects str for symbolic params
                    else:
                        values.append(0.0)
                else:
                    values.append(float(expr_eval.evaluate(child)))
        return values

    def _build_qref(self, node: Tree) -> QRef:
        name, index, line, col = self._extract_reference(node)
        if self._gate_context:
            ctx = self._gate_context[-1]
            if name not in ctx.qargs:
                raise QasmError("E402", f"Unknown gate argument '{name}' in gate '{ctx.name}'.", line, col)
            return QRef(reg=name, idx=index, line=line, col=col)
        size = self._qreg_sizes.get(name)
        if size is None:
            raise QasmError("E402", f"Unknown quantum register '{name}'.", line, col)
        if index is not None and (index < 0 or index >= size):
            raise QasmError("E402", f"Qubit index {index} out of range for register '{name}'.", line, col)
        return QRef(reg=name, idx=index, line=line, col=col)

    def _build_cref(self, node: Tree) -> CRef:
        name, index, line, col = self._extract_reference(node)
        size = self._creg_sizes.get(name)
        if size is None:
            raise QasmError("E402", f"Unknown classical register '{name}'.", line, col)
        if index is not None and (index < 0 or index >= size):
            raise QasmError("E402", f"Classical index {index} out of range for register '{name}'.", line, col)
        return CRef(reg=name, idx=index, line=line, col=col)

    def _extract_reference(self, node: Tree) -> tuple[str, Optional[int], int, int]:
        name_token = _find_token(node.children, "CNAME")
        index_token = _find_token(node.children, "INT")
        if name_token is None:
            line, col = _node_location(node)
            raise QasmError("E401", "Missing identifier in reference.", line, col)
        index = int(index_token.value) if index_token is not None else None
        line, col = _node_location(node)
        return name_token.value, index, line, col

    def _ensure_gate_known(self, name: str, line: int, col: int) -> None:
        if name in self._BUILTIN_GATES:
            return
        if name in self._gate_defs:
            return
        raise QasmError("E402", f"Unknown gate '{name}'.", line, col)


def _find_token(children: Iterable[object], token_type: str) -> Optional[Token]:
    for child in children:
        if isinstance(child, Token) and child.type == token_type:
            return child
        if isinstance(child, Tree):
            token = _find_token(child.children, token_type)
            if token is not None:
                return token
    return None


def _find_tree(children: Iterable[object], names: set[str]) -> Optional[Tree]:
    for child in children:
        if isinstance(child, Tree):
            if child.data in names:
                return child
            nested = _find_tree(child.children, names)
            if nested is not None:
                return nested
    return None


def _find_first_identifier(children: Iterable[object]) -> Optional[Token]:
    for child in children:
        if isinstance(child, Token) and child.type not in {"INT", "NUMBER", "SIGNED_NUMBER"}:
            return child
        if isinstance(child, Tree):
            token = _find_first_identifier(child.children)
            if token is not None:
                return token
    return None


def _node_location(node: Tree | Token) -> tuple[int, int]:
    if isinstance(node, Tree):
        meta = getattr(node, "meta", None)
        if meta is not None:
            line = getattr(meta, "line", None)
            column = getattr(meta, "column", None)
            if line is not None and column is not None:
                return int(line), int(column)
        if node.children:
            for child in node.children:
                if isinstance(child, (Tree, Token)):
                    return _node_location(child)
        return 1, 1
    line = getattr(node, "line", None)
    column = getattr(node, "column", None)
    if line is not None and column is not None:
        return int(line), int(column)
    return 1, 1


@lru_cache(maxsize=1)
def create_parser() -> Lark:
    """Instantiate the Lark parser for the strict OpenQASM grammar.

    Returns
    -------
    Lark
        Configured parser instance with position propagation enabled.
    """
    try:
        # Python 3.9+ importlib.resources API
        grammar_text = importlib_resources.files(_GRAMMAR_PACKAGE).joinpath(_GRAMMAR_FILE).read_text(encoding="utf-8")
    except (AttributeError, FileNotFoundError, ModuleNotFoundError):
        try:
            grammar_text = importlib_resources.read_text(_GRAMMAR_PACKAGE, _GRAMMAR_FILE, encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError, UnicodeDecodeError):
            grammar_path = Path(__file__).with_name("grammar").joinpath(_GRAMMAR_FILE)
            grammar_text = grammar_path.read_text(encoding="utf-8")
    return Lark(
        grammar_text,
        start="start",
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


def parse_qasm(text: str) -> ProgramAST:
    """Parse OpenQASM 2 source into a typed AST.

    Parameters
    ----------
    text : str
        OpenQASM 2 source code.

    Returns
    -------
    ProgramAST
        Structured representation of the program.

    Raises
    ------
    QasmError
        If parsing fails or semantic checks detect inconsistencies.
    """
    parser = create_parser()
    try:
        tree = parser.parse(text)
    except UnexpectedInput as exc:
        line = getattr(exc, "line", 1) or 1
        column = getattr(exc, "column", 1) or 1
        message = getattr(exc, "msg", None) or "Failed to parse OpenQASM source."
        raise QasmError("E401", message, line, column) from exc
    except LarkError as exc:
        raise QasmError("E401", "Failed to parse OpenQASM source.", 1, 1) from exc

    transformer = QasmTransformer()
    return transformer.transform(tree)


def parse_qasm_file(file_path: str) -> ProgramAST:
    """Parse a file containing OpenQASM 2 source.

    Parameters
    ----------
    file_path : str
        Path to the file that should be parsed.

    Returns
    -------
    ProgramAST
        Structured representation identical to :func:`parse_qasm`.
    """
    source_path = Path(file_path)
    text = source_path.read_text(encoding="utf-8")
    return parse_qasm(text)
