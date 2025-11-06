"""Safe evaluation utilities for OpenQASM 2 arithmetic expressions."""

from __future__ import annotations

import math
import operator
from typing import Callable

from lark import Tree, Token

from qasm2.errors import QasmError

__all__ = ["ALLOWED_CONSTANTS", "ALLOWED_FUNCS", "evaluate"]

ALLOWED_CONSTANTS: dict[str, float] = {"pi": float(math.pi)}

ALLOWED_FUNCS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}

_BINARY_OPERATORS: dict[str, Callable[[float, float], float]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
}

_UNARY_OPERATORS: set[str] = {"neg", "unary_minus"}


def _node_location(node: Tree | Token) -> tuple[int, int]:
    """Best-effort extraction of source location metadata."""
    line = getattr(node, "line", None)
    col = getattr(node, "column", None)
    if line is not None and col is not None:
        return int(line), int(col)

    if isinstance(node, Tree):
        meta = getattr(node, "meta", None)
        if meta is not None:
            meta_line = getattr(meta, "line", None)
            meta_col = getattr(meta, "column", None)
            if meta_line is not None and meta_col is not None:
                return int(meta_line), int(meta_col)
        for child in node.children:
            if isinstance(child, (Tree, Token)):
                child_line, child_col = _node_location(child)
                if child_line is not None and child_col is not None:
                    return child_line, child_col

    return 1, 1


def _make_error(node: Tree | Token, message: str) -> QasmError:
    """Build a QasmError anchored at the supplied node."""
    line, col = _node_location(node)
    return QasmError("E301", message, line, col)


def _parse_number(node: Tree | Token) -> float:
    """Parse a numeric literal token into a float value."""
    token = node
    if isinstance(node, Tree):
        if len(node.children) != 1:
            raise _make_error(node, "Malformed numeric literal.")
        child = node.children[0]
        if not isinstance(child, Token):
            raise _make_error(node, "Numeric literal does not contain a token.")
        token = child

    if not isinstance(token, Token):
        raise _make_error(node, "Expected numeric token.")

    try:
        return float(token.value)
    except ValueError as exc:
        raise _make_error(token, f"Invalid numeric literal '{token.value}'.") from exc


def evaluate(tree: Tree | Token) -> float:
    """Evaluate a parsed OpenQASM expression tree.

    Parameters
    ----------
    tree : Tree | Token
        Parse tree or token produced by the OpenQASM expression grammar.

    Returns
    -------
    float
        Evaluated numeric value expressed in radians (float64).

    Raises
    ------
    QasmError
        If the expression contains unsupported constructs or malformed nodes.
    """
    if isinstance(tree, Token):
        if tree.type in {"NUMBER", "INT", "SIGNED_NUMBER"}:
            return float(tree.value)
        if tree.type == "CNAME" and tree.value in ALLOWED_CONSTANTS:
            return float(ALLOWED_CONSTANTS[tree.value])
        raise _make_error(tree, f"Unexpected token '{tree.type}' in expression.")

    if not isinstance(tree, Tree):
        raise TypeError(f"Unsupported node type: {type(tree)!r}")

    data = tree.data
    children = tree.children

    if data == "number":
        return _parse_number(tree)

    if data == "pi":
        return float(ALLOWED_CONSTANTS["pi"])

    if data in _BINARY_OPERATORS:
        if len(children) != 2:
            raise _make_error(tree, f"Operator '{data}' requires exactly two operands.")
        left_child = children[0]
        right_child = children[1]
        if not isinstance(left_child, (Tree, Token)) or not isinstance(right_child, (Tree, Token)):
            raise _make_error(tree, "Binary operator requires Tree or Token children.")
        left = evaluate(left_child)
        right = evaluate(right_child)
        try:
            result = _BINARY_OPERATORS[data](left, right)
        except ZeroDivisionError as exc:
            raise _make_error(right_child, "Division by zero in expression.") from exc
        return float(result)

    if data in _UNARY_OPERATORS:
        if len(children) != 1:
            raise _make_error(tree, "Unary minus requires exactly one operand.")
        child = children[0]
        if not isinstance(child, (Tree, Token)):
            raise _make_error(tree, "Unary operator requires Tree or Token child.")
        return float(-evaluate(child))

    if data == "fun1":
        if len(children) != 2:
            raise _make_error(tree, "Malformed function call.")
        func_token, arg_tree = children
        if not isinstance(func_token, Token):
            raise _make_error(tree, "Function identifier must be a token.")
        if not isinstance(arg_tree, (Tree, Token)):
            raise _make_error(tree, "Function argument must be Tree or Token.")
        func_name = func_token.value
        func = ALLOWED_FUNCS.get(func_name)
        if func is None:
            raise _make_error(func_token, f"Function '{func_name}' is not permitted.")
        argument = evaluate(arg_tree)
        return float(func(argument))

    if data in {"expr", "sum", "product", "power", "atom"}:
        if len(children) != 1:
            raise _make_error(tree, f"Expression node '{data}' must have a single child.")
        child = children[0]
        if not isinstance(child, (Tree, Token)):
            raise _make_error(tree, "Expression node requires Tree or Token child.")
        return evaluate(child)

    raise _make_error(tree, f"Unsupported expression node '{data}'.")
