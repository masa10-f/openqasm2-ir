API リファレンス
================

API Reference
=============

このページでは、qasm2graphqombの主要なAPIを説明します。

This page describes the main APIs of qasm2graphqomb.

パーサーモジュール / Parser Module
----------------------------------

.. automodule:: qasm2.parser
   :members:
   :undoc-members:
   :show-inheritance:

主な関数 / Main Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qasm2.parser.parse_qasm

.. autofunction:: qasm2.parser.parse_qasm_file

.. autofunction:: qasm2.parser.create_parser

トランスフォーマー / Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qasm2.parser.QasmTransformer
   :members:
   :undoc-members:
   :show-inheritance:

中間表現モジュール / IR Module
------------------------------

.. automodule:: ir.circuit
   :members:
   :undoc-members:
   :show-inheritance:

Circuit クラス / Circuit Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ir.circuit.Circuit
   :members:
   :undoc-members:
   :show-inheritance:

Op クラス / Op Class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ir.circuit.Op
   :members:
   :undoc-members:
   :show-inheritance:

ローワリングモジュール / Lowering Module
----------------------------------------

.. automodule:: qasm2.lower
   :members:
   :undoc-members:
   :show-inheritance:

正規化モジュール / Normalization Module
---------------------------------------

.. automodule:: qasm2.normalize
   :members:
   :undoc-members:
   :show-inheritance:

検証モジュール / Validation Module
----------------------------------

.. automodule:: qasm2.validate
   :members:
   :undoc-members:
   :show-inheritance:

ASTノードモジュール / AST Nodes Module
--------------------------------------

.. automodule:: qasm2.ast_nodes
   :members:
   :undoc-members:
   :show-inheritance:

式評価モジュール / Expression Evaluation Module
-----------------------------------------------

.. automodule:: qasm2.expr_eval
   :members:
   :undoc-members:
   :show-inheritance:

エラー処理モジュール / Error Handling Module
--------------------------------------------

.. automodule:: qasm2.errors
   :members:
   :undoc-members:
   :show-inheritance:

QasmError クラス / QasmError Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: qasm2.errors.QasmError
   :members:
   :undoc-members:
   :show-inheritance:

GraphQOMBアダプターモジュール / GraphQOMB Adapter Module
-------------------------------------------------------

.. automodule:: graphqomb_adapter.converter
   :members:
   :undoc-members:
   :show-inheritance:

CLIモジュール / CLI Module
--------------------------

.. automodule:: qasm2.cli
   :members:
   :undoc-members:
   :show-inheritance:
