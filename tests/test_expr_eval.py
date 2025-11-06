"""Test suite for qasm2.expr_eval module.

This module tests the safe evaluation utilities for OpenQASM 2 arithmetic expressions,
covering constants, operators, functions, and error handling.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from lark import Tree, Token  # type: ignore[import-untyped]

from qasm2.errors import QasmError
from qasm2.expr_eval import ALLOWED_CONSTANTS, ALLOWED_FUNCS, evaluate


class TestConstants:
    """Test evaluation of mathematical constants."""

    def test_pi_constant(self) -> None:
        """Test evaluation of pi constant."""
        token = Token("CNAME", "pi")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == pytest.approx(math.pi)

    def test_pi_in_allowed_constants(self) -> None:
        """Verify pi is in ALLOWED_CONSTANTS dictionary."""
        assert "pi" in ALLOWED_CONSTANTS
        assert ALLOWED_CONSTANTS["pi"] == pytest.approx(math.pi)


class TestNumericLiterals:
    """Test evaluation of numeric literals."""

    def test_integer_token(self) -> None:
        """Test evaluation of integer token."""
        token = Token("INT", "42")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == 42.0

    def test_float_token(self) -> None:
        """Test evaluation of floating-point number token."""
        token = Token("NUMBER", "3.14159")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == pytest.approx(3.14159)

    def test_signed_number_token(self) -> None:
        """Test evaluation of signed number token."""
        token = Token("SIGNED_NUMBER", "-2.5")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == -2.5

    def test_number_tree(self) -> None:
        """Test evaluation of number tree node."""
        token = Token("NUMBER", "1.5")  # type: ignore[arg-type]
        tree = Tree("number", [token])
        result = evaluate(tree)
        assert result == 1.5


class TestBinaryOperators:
    """Test evaluation of binary operators."""

    def test_addition(self) -> None:
        """Test addition operator."""
        left = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("add", [left, right])
        result = evaluate(tree)
        assert result == 3.0

    def test_subtraction(self) -> None:
        """Test subtraction operator."""
        left = Token("NUMBER", "5.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        tree = Tree("sub", [left, right])
        result = evaluate(tree)
        assert result == 2.0

    def test_multiplication(self) -> None:
        """Test multiplication operator."""
        left = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "2.5")  # type: ignore[arg-type]
        tree = Tree("mul", [left, right])
        result = evaluate(tree)
        assert result == 10.0

    def test_division(self) -> None:
        """Test division operator."""
        left = Token("NUMBER", "10.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        tree = Tree("div", [left, right])
        result = evaluate(tree)
        assert result == 2.5

    def test_power(self) -> None:
        """Test power operator (exponentiation)."""
        left = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        tree = Tree("pow", [left, right])
        result = evaluate(tree)
        assert result == 8.0

    def test_division_by_zero(self) -> None:
        """Test that division by zero raises QasmError."""
        left = Token("NUMBER", "10.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("div", [left, right])
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "Division by zero" in str(exc_info.value)


class TestUnaryOperators:
    """Test evaluation of unary operators."""

    def test_unary_minus_neg(self) -> None:
        """Test unary minus with 'neg' node."""
        operand = Token("NUMBER", "5.0")  # type: ignore[arg-type]
        tree = Tree("neg", [operand])
        result = evaluate(tree)
        assert result == -5.0

    def test_unary_minus_explicit(self) -> None:
        """Test unary minus with 'unary_minus' node."""
        operand = Token("NUMBER", "3.14")  # type: ignore[arg-type]
        tree = Tree("unary_minus", [operand])
        result = evaluate(tree)
        assert result == pytest.approx(-3.14)

    def test_double_negation(self) -> None:
        """Test double negation."""
        inner_operand = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        inner_tree = Tree("neg", [inner_operand])
        outer_tree = Tree("neg", [inner_tree])
        result = evaluate(outer_tree)
        assert result == 2.0


class TestMathFunctions:
    """Test evaluation of mathematical functions."""

    def test_sin_function(self) -> None:
        """Test sin function."""
        func_name = Token("CNAME", "sin")  # type: ignore[arg-type]
        arg = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(0.0)

    def test_sin_pi_over_2(self) -> None:
        """Test sin(pi/2) = 1."""
        func_name = Token("CNAME", "sin")  # type: ignore[arg-type]
        pi_token = Token("CNAME", "pi")  # type: ignore[arg-type]
        two_token = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        div_tree = Tree("div", [pi_token, two_token])
        tree = Tree("fun1", [func_name, div_tree])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)

    def test_cos_function(self) -> None:
        """Test cos function."""
        func_name = Token("CNAME", "cos")  # type: ignore[arg-type]
        arg = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)

    def test_cos_pi(self) -> None:
        """Test cos(pi) = -1."""
        func_name = Token("CNAME", "cos")  # type: ignore[arg-type]
        arg = Token("CNAME", "pi")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(-1.0)

    def test_tan_function(self) -> None:
        """Test tan function."""
        func_name = Token("CNAME", "tan")  # type: ignore[arg-type]
        arg = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(0.0)

    def test_exp_function(self) -> None:
        """Test exp function."""
        func_name = Token("CNAME", "exp")  # type: ignore[arg-type]
        arg = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(math.e)

    def test_ln_function(self) -> None:
        """Test ln (natural logarithm) function."""
        func_name = Token("CNAME", "ln")  # type: ignore[arg-type]
        arg = Token("NUMBER", str(math.e))  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)

    def test_sqrt_function(self) -> None:
        """Test sqrt function."""
        func_name = Token("CNAME", "sqrt")  # type: ignore[arg-type]
        arg = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(2.0)

    def test_sqrt_two(self) -> None:
        """Test sqrt(2)."""
        func_name = Token("CNAME", "sqrt")  # type: ignore[arg-type]
        arg = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(math.sqrt(2))

    def test_all_functions_in_allowed_funcs(self) -> None:
        """Verify all expected functions are in ALLOWED_FUNCS."""
        expected_funcs = {"sin", "cos", "tan", "exp", "ln", "sqrt"}
        assert set(ALLOWED_FUNCS.keys()) == expected_funcs


class TestComplexExpressions:
    """Test evaluation of complex nested expressions."""

    def test_pi_over_2(self) -> None:
        """Test pi/2 expression."""
        pi_token = Token("CNAME", "pi")  # type: ignore[arg-type]
        two_token = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("div", [pi_token, two_token])
        result = evaluate(tree)
        assert result == pytest.approx(math.pi / 2)

    def test_sin_pi_over_4(self) -> None:
        """Test sin(pi/4) = sqrt(2)/2."""
        func_name = Token("CNAME", "sin")  # type: ignore[arg-type]
        pi_token = Token("CNAME", "pi")  # type: ignore[arg-type]
        four_token = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        div_tree = Tree("div", [pi_token, four_token])
        tree = Tree("fun1", [func_name, div_tree])
        result = evaluate(tree)
        assert result == pytest.approx(math.sin(math.pi / 4))

    def test_negative_three_pi_over_4(self) -> None:
        """Test -3*pi/4 expression."""
        three_token = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        pi_token = Token("CNAME", "pi")  # type: ignore[arg-type]
        mul_tree = Tree("mul", [three_token, pi_token])
        four_token = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        div_tree = Tree("div", [mul_tree, four_token])
        neg_tree = Tree("neg", [div_tree])
        result = evaluate(neg_tree)
        assert result == pytest.approx(-3 * math.pi / 4)

    def test_sqrt_two_over_2(self) -> None:
        """Test sqrt(2)/2 expression."""
        func_name = Token("CNAME", "sqrt")  # type: ignore[arg-type]
        two_token_1 = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        sqrt_tree = Tree("fun1", [func_name, two_token_1])
        two_token_2 = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("div", [sqrt_tree, two_token_2])
        result = evaluate(tree)
        assert result == pytest.approx(math.sqrt(2) / 2)

    def test_nested_operations(self) -> None:
        """Test deeply nested arithmetic operations: (1 + 2) * (3 - 4)."""
        one = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        two = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        add_tree = Tree("add", [one, two])
        three = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        four = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        sub_tree = Tree("sub", [three, four])
        mul_tree = Tree("mul", [add_tree, sub_tree])
        result = evaluate(mul_tree)
        assert result == pytest.approx((1 + 2) * (3 - 4))

    def test_power_with_expression(self) -> None:
        """Test 2^(1+2) = 8."""
        base = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        one = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        two = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        add_tree = Tree("add", [one, two])
        pow_tree = Tree("pow", [base, add_tree])
        result = evaluate(pow_tree)
        assert result == pytest.approx(8.0)

    def test_exp_ln_identity(self) -> None:
        """Test exp(ln(x)) = x for x = 5."""
        x_value = "5.0"
        ln_func = Token("CNAME", "ln")  # type: ignore[arg-type]
        x_token = Token("NUMBER", x_value)  # type: ignore[arg-type]
        ln_tree = Tree("fun1", [ln_func, x_token])
        exp_func = Token("CNAME", "exp")  # type: ignore[arg-type]
        exp_tree = Tree("fun1", [exp_func, ln_tree])
        result = evaluate(exp_tree)
        assert result == pytest.approx(5.0)


class TestExpressionWrappers:
    """Test evaluation of expression wrapper nodes."""

    def test_expr_wrapper(self) -> None:
        """Test 'expr' wrapper node."""
        token = Token("NUMBER", "42.0")  # type: ignore[arg-type]
        tree = Tree("expr", [token])
        result = evaluate(tree)
        assert result == 42.0

    def test_sum_wrapper(self) -> None:
        """Test 'sum' wrapper node."""
        left = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        add_tree = Tree("add", [left, right])
        sum_tree = Tree("sum", [add_tree])
        result = evaluate(sum_tree)
        assert result == 3.0

    def test_product_wrapper(self) -> None:
        """Test 'product' wrapper node."""
        left = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        right = Token("NUMBER", "4.0")  # type: ignore[arg-type]
        mul_tree = Tree("mul", [left, right])
        product_tree = Tree("product", [mul_tree])
        result = evaluate(product_tree)
        assert result == 12.0

    def test_power_wrapper(self) -> None:
        """Test 'power' wrapper node."""
        base = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        exp = Token("NUMBER", "3.0")  # type: ignore[arg-type]
        pow_tree = Tree("pow", [base, exp])
        power_tree = Tree("power", [pow_tree])
        result = evaluate(power_tree)
        assert result == 8.0

    def test_atom_wrapper(self) -> None:
        """Test 'atom' wrapper node."""
        token = Token("NUMBER", "7.5")  # type: ignore[arg-type]
        atom_tree = Tree("atom", [token])
        result = evaluate(atom_tree)
        assert result == 7.5

    def test_nested_wrappers(self) -> None:
        """Test nested wrapper nodes."""
        token = Token("NUMBER", "10.0")  # type: ignore[arg-type]
        atom_tree = Tree("atom", [token])
        expr_tree = Tree("expr", [atom_tree])
        result = evaluate(expr_tree)
        assert result == 10.0


class TestErrorCases:
    """Test error handling for invalid expressions."""

    def test_unsupported_function(self) -> None:
        """Test that unsupported function raises QasmError."""
        func_name = Token("CNAME", "log10")  # type: ignore[arg-type]  # Not in ALLOWED_FUNCS
        arg = Token("NUMBER", "10.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "log10" in str(exc_info.value)
        assert "not permitted" in str(exc_info.value)

    def test_undefined_constant(self) -> None:
        """Test that undefined constant raises QasmError."""
        token = Token("CNAME", "euler")  # type: ignore[arg-type]  # Not in ALLOWED_CONSTANTS
        with pytest.raises(QasmError) as exc_info:
            evaluate(token)
        assert "E301" in str(exc_info.value)

    def test_invalid_numeric_literal(self) -> None:
        """Test that invalid numeric literal raises QasmError."""
        token = Token("NUMBER", "not_a_number")  # type: ignore[arg-type]
        tree = Tree("number", [token])
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "Invalid numeric literal" in str(exc_info.value)

    def test_unsupported_node_type(self) -> None:
        """Test that unsupported node type raises QasmError."""
        tree = Tree("unknown_operation", [Token("NUMBER", "1.0")])  # type: ignore[arg-type]
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "Unsupported expression node" in str(exc_info.value)

    def test_unexpected_token_type(self) -> None:
        """Test that unexpected token type raises QasmError."""
        token = Token("UNEXPECTED_TYPE", "value")  # type: ignore[arg-type]
        with pytest.raises(QasmError) as exc_info:
            evaluate(token)
        assert "E301" in str(exc_info.value)
        assert "Unexpected token" in str(exc_info.value)

    def test_binary_operator_missing_operand(self) -> None:
        """Test that binary operator with missing operand raises QasmError."""
        operand = Token("NUMBER", "5.0")  # type: ignore[arg-type]
        tree = Tree("add", [operand])  # Missing second operand
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "requires exactly two operands" in str(exc_info.value)

    def test_unary_operator_missing_operand(self) -> None:
        """Test that unary operator with no operand raises QasmError."""
        tree = Tree("neg", [])  # No operand
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "requires exactly one operand" in str(exc_info.value)

    def test_function_malformed(self) -> None:
        """Test that malformed function call raises QasmError."""
        func_name = Token("CNAME", "sin")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name])  # Missing argument
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "Malformed function call" in str(exc_info.value)

    def test_wrapper_multiple_children(self) -> None:
        """Test that wrapper node with multiple children raises QasmError."""
        token1 = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        token2 = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("expr", [token1, token2])  # Should have single child
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "must have a single child" in str(exc_info.value)

    def test_number_tree_malformed(self) -> None:
        """Test that malformed number tree raises QasmError."""
        token1 = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        token2 = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        tree = Tree("number", [token1, token2])  # Should have single child
        with pytest.raises(QasmError) as exc_info:
            evaluate(tree)
        assert "E301" in str(exc_info.value)
        assert "Malformed numeric literal" in str(exc_info.value)

    def test_invalid_type_raises_typeerror(self) -> None:
        """Test that passing invalid type raises TypeError."""
        invalid_input: Any = "not a tree or token"
        with pytest.raises(TypeError) as exc_info:
            evaluate(invalid_input)
        assert "Unsupported node type" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_operations(self) -> None:
        """Test operations involving zero."""
        zero = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        result = evaluate(zero)
        assert result == 0.0

    def test_very_small_number(self) -> None:
        """Test very small floating-point number."""
        token = Token("NUMBER", "1e-100")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == pytest.approx(1e-100)

    def test_very_large_number(self) -> None:
        """Test very large floating-point number."""
        token = Token("NUMBER", "1e100")  # type: ignore[arg-type]
        result = evaluate(token)
        assert result == pytest.approx(1e100)

    def test_negative_zero(self) -> None:
        """Test negative zero."""
        zero = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("neg", [zero])
        result = evaluate(tree)
        assert result == 0.0  # -0.0 == 0.0 in Python

    def test_pi_tree_node(self) -> None:
        """Test 'pi' as Tree node (not just token)."""
        tree = Tree("pi", [])
        result = evaluate(tree)
        assert result == pytest.approx(math.pi)

    def test_integer_division_returns_float(self) -> None:
        """Test that integer division returns float result."""
        left = Token("INT", "7")  # type: ignore[arg-type]
        right = Token("INT", "2")  # type: ignore[arg-type]
        tree = Tree("div", [left, right])
        result = evaluate(tree)
        assert isinstance(result, float)
        assert result == 3.5

    def test_power_zero_exponent(self) -> None:
        """Test x^0 = 1."""
        base = Token("NUMBER", "42.0")  # type: ignore[arg-type]
        exp = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("pow", [base, exp])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)

    def test_power_negative_exponent(self) -> None:
        """Test 2^(-1) = 0.5."""
        base = Token("NUMBER", "2.0")  # type: ignore[arg-type]
        exp = Token("NUMBER", "-1.0")  # type: ignore[arg-type]
        tree = Tree("pow", [base, exp])
        result = evaluate(tree)
        assert result == pytest.approx(0.5)

    def test_sqrt_one(self) -> None:
        """Test sqrt(1) = 1."""
        func_name = Token("CNAME", "sqrt")  # type: ignore[arg-type]
        arg = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)

    def test_ln_one(self) -> None:
        """Test ln(1) = 0."""
        func_name = Token("CNAME", "ln")  # type: ignore[arg-type]
        arg = Token("NUMBER", "1.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(0.0)

    def test_exp_zero(self) -> None:
        """Test exp(0) = 1."""
        func_name = Token("CNAME", "exp")  # type: ignore[arg-type]
        arg = Token("NUMBER", "0.0")  # type: ignore[arg-type]
        tree = Tree("fun1", [func_name, arg])
        result = evaluate(tree)
        assert result == pytest.approx(1.0)
