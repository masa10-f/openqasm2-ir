チュートリアル
==============

このチュートリアルでは、qasm2graphqombの基本的な使い方から応用まで、ステップバイステップで解説します。

Tutorial
========

This tutorial provides step-by-step guidance on using qasm2graphqomb, from basics to advanced usage.

基本的なワークフロー / Basic Workflow
-------------------------------------

1. OpenQASM 2.0のパース / Parsing OpenQASM 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenQASM 2.0のソースコードをパースして、ASTを取得します。

Parse OpenQASM 2.0 source code to obtain an AST.

.. code-block:: python

   from qasm2.parser import parse_qasm, parse_qasm_file

   # 文字列からパース / Parse from string
   qasm_code = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   creg c[2];
   h q[0];
   cx q[0], q[1];
   measure q -> c;
   """
   ast = parse_qasm(qasm_code)

   # ファイルからパース / Parse from file
   ast = parse_qasm_file("circuit.qasm")

   # ASTの内容を確認 / Inspect AST contents
   print(f"OpenQASM version: {ast.version}")
   print(f"Quantum registers: {ast.qregs}")
   print(f"Classical registers: {ast.cregs}")
   print(f"Gate definitions: {list(ast.gate_defs.keys())}")
   print(f"Operations: {len(ast.body)}")

2. 中間表現（IR）への変換 / Converting to IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASTを効率的な中間表現（IR）に変換します。

Convert the AST to an efficient intermediate representation (IR).

.. code-block:: python

   from qasm2.lower import lower_program

   # ASTをIRに変換 / Convert AST to IR
   circuit = lower_program(ast)

   # 回路の情報を取得 / Get circuit information
   print(f"Number of qubits: {circuit.n_qubits}")
   print(f"Number of operations: {len(circuit.ops)}")

   # 各操作の詳細を確認 / Inspect operation details
   for i, op in enumerate(circuit.ops):
       print(f"Op {i}: {op.name} on qubits {op.qubits} with params {op.params}")

3. 回路の保存と読み込み / Saving and Loading Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IRをJSON形式で保存・読み込みできます。

You can save and load IR in JSON format.

.. code-block:: python

   from ir.circuit import Circuit

   # JSON形式で保存 / Save as JSON
   circuit.to_json("circuit.json")

   # JSONから読み込み / Load from JSON
   loaded_circuit = Circuit.from_json("circuit.json")

   # 辞書形式でも扱える / Can also work with dictionaries
   circuit_dict = circuit.to_dict()
   restored_circuit = Circuit.from_dict(circuit_dict)

実践例 / Practical Examples
---------------------------

ベル状態の生成 / Creating a Bell State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2量子ビットのベル状態を生成する回路を作成します。

Create a circuit that generates a two-qubit Bell state.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.lower import lower_program

   # ベル状態を生成するOpenQASMコード
   # OpenQASM code to generate a Bell state
   bell_state_qasm = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   creg c[2];
   h q[0];
   cx q[0], q[1];
   measure q -> c;
   """

   # パースと変換 / Parse and convert
   ast = parse_qasm(bell_state_qasm)
   circuit = lower_program(ast)

   print("Bell State Circuit:")
   print(f"  Qubits: {circuit.n_qubits}")
   for op in circuit.ops:
       print(f"  {op.name} on qubits {op.qubits}")

量子フーリエ変換 / Quantum Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3量子ビットの量子フーリエ変換を実装します。

Implement a 3-qubit Quantum Fourier Transform.

.. code-block:: python

   qft3_qasm = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[3];

   // QFT on 3 qubits
   h q[0];
   p(pi/2) q[0];
   cx q[1], q[0];
   h q[1];
   p(pi/4) q[0];
   p(pi/2) q[1];
   cx q[2], q[0];
   cx q[2], q[1];
   h q[2];

   // Swap
   swap q[0], q[2];
   """

   ast = parse_qasm(qft3_qasm)
   circuit = lower_program(ast)

   print("QFT Circuit:")
   print(f"  Total operations: {len(circuit.ops)}")

カスタムゲートの定義 / Defining Custom Gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

独自のゲートを定義して使用します。

Define and use custom gates.

.. code-block:: python

   custom_gate_qasm = """
   OPENQASM 2.0;
   include "qelib1.inc";

   // カスタムゲートの定義 / Define custom gate
   gate my_gate(theta, phi) q {
       rx(theta) q;
       ry(phi) q;
   }

   qreg q[2];

   // カスタムゲートの使用 / Use custom gate
   my_gate(pi/4, pi/2) q[0];
   my_gate(pi/3, pi/6) q[1];
   """

   ast = parse_qasm(custom_gate_qasm)
   print(f"Defined gates: {list(ast.gate_defs.keys())}")

   circuit = lower_program(ast)
   for op in circuit.ops:
       print(f"{op.name}: qubits={op.qubits}, params={op.params}")

GraphQOMBへの変換 / Converting to GraphQOMB
-------------------------------------------

GraphQOMB形式への変換を行います。

Convert to GraphQOMB format.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.lower import lower_program
   from graphqomb_adapter.converter import circuit_to_graphqomb

   # 量子回路をパースしてIRに変換
   # Parse quantum circuit and convert to IR
   qasm_code = """
   OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[3];
   h q[0];
   cx q[0], q[1];
   cx q[1], q[2];
   """

   ast = parse_qasm(qasm_code)
   circuit = lower_program(ast)

   # GraphQOMB形式に変換 / Convert to GraphQOMB format
   graphqomb_circuit = circuit_to_graphqomb(circuit)

   print(f"GraphQOMB circuit created with {graphqomb_circuit.nqubits} qubits")

コマンドラインツールの使用 / Using CLI Tools
--------------------------------------------

コマンドラインから直接変換を行うこともできます。

You can also perform conversions directly from the command line.

.. code-block:: bash

   # OpenQASMファイルをJSON形式のIRに変換
   # Convert OpenQASM file to JSON IR
   qasm-import input.qasm -o output.json

   # 標準出力に出力
   # Output to stdout
   qasm-import input.qasm

エラーハンドリング / Error Handling
-----------------------------------

パースエラーや検証エラーを適切に処理します。

Handle parsing and validation errors appropriately.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.errors import QasmError

   invalid_qasm = """
   OPENQASM 2.0;
   qreg q[2];
   unknown_gate q[0];  // 未定義のゲート / Undefined gate
   """

   try:
       ast = parse_qasm(invalid_qasm)
   except QasmError as e:
       print(f"Error {e.code} at line {e.line}, column {e.column}:")
       print(f"  {e.message}")

高度な使い方 / Advanced Usage
----------------------------

回路の検証 / Circuit Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

正規化と検証を行います。

Perform normalization and validation.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.normalize import normalize_program
   from qasm2.validate import validate_program

   ast = parse_qasm(qasm_code)

   # 正規化 / Normalize
   normalized_ast = normalize_program(ast)

   # 検証 / Validate
   validate_program(normalized_ast)

回路の操作 / Circuit Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IR形式で回路を操作します。

Manipulate circuits in IR format.

.. code-block:: python

   from ir.circuit import Circuit, Op
   import math

   # 新しい回路を作成 / Create a new circuit
   ops = [
       Op(name="h", qubits=[0], params=()),
       Op(name="rx", qubits=[0], params=(math.pi/4,)),
       Op(name="cx", qubits=[0, 1], params=()),
   ]

   circuit = Circuit(
       n_qubits=2,
       ops=ops,
       meas_map=[(0, 0), (1, 1)]
   )

   # 操作の追加 / Add operations
   new_ops = circuit.ops + [Op(name="h", qubits=[1], params=())]
   modified_circuit = Circuit(
       n_qubits=circuit.n_qubits,
       ops=new_ops,
       meas_map=circuit.meas_map
   )

まとめ / Summary
---------------

このチュートリアルでは以下を学びました：

This tutorial covered:

1. OpenQASM 2.0のパースと検証 / Parsing and validating OpenQASM 2.0
2. 中間表現（IR）への変換 / Converting to Intermediate Representation (IR)
3. 回路の保存と読み込み / Saving and loading circuits
4. GraphQOMBへの変換 / Converting to GraphQOMB
5. コマンドラインツールの使用 / Using CLI tools
6. エラーハンドリングと高度な操作 / Error handling and advanced operations

さらに詳しい情報は :doc:`api` を参照してください。

For more detailed information, see the :doc:`api` reference.
