qasm2graphqomb ドキュメント
================================

**qasm2graphqomb** は、OpenQASM 2.0のソースコードを解析し、カスタム中間表現（IR）に変換するPythonライブラリです。
量子回路の操作、検証、GraphQOMBとの統合をサポートします。

.. toctree::
   :maxdepth: 2
   :caption: 目次:

   tutorial
   api
   modules

主な機能
--------

* **OpenQASM 2.0パーサー**: 完全なOpenQASM 2.0構文サポート
* **中間表現（IR）**: 量子回路の効率的な表現と操作
* **GraphQOMB統合**: GraphQOMBへのシームレスな変換
* **型安全性**: 静的型チェックによる信頼性の高い開発
* **CLIツール**: コマンドラインからの簡単な利用

クイックスタート
----------------

インストール
~~~~~~~~~~~~

.. code-block:: bash

   pip install qasm2graphqomb

基本的な使い方
~~~~~~~~~~~~~~

OpenQASM 2.0ファイルをパースして中間表現に変換：

.. code-block:: python

   from qasm2.parser import parse_qasm_file
   from qasm2.lower import lower_program

   # OpenQASMファイルをパース
   ast = parse_qasm_file("circuit.qasm")

   # 中間表現（IR）に変換
   circuit = lower_program(ast)

   # 量子回路の情報を取得
   print(f"量子ビット数: {circuit.n_qubits}")
   print(f"操作数: {len(circuit.ops)}")

コマンドラインツール
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # OpenQASMファイルをJSON形式のIRに変換
   qasm-import input.qasm -o output.json

インデックスと検索
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
