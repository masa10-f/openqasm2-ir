API Reference
=============

This page describes the main APIs of qasm2graphqomb.

Parser Module
-------------

.. automodule:: qasm2.parser
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
~~~~~~~~~~~~~~

.. autofunction:: qasm2.parser.parse_qasm

.. autofunction:: qasm2.parser.parse_qasm_file

.. autofunction:: qasm2.parser.create_parser

Transformer
~~~~~~~~~~~

.. autoclass:: qasm2.parser.QasmTransformer
   :members:
   :undoc-members:
   :show-inheritance:

IR Module
---------

.. automodule:: ir.circuit
   :members:
   :undoc-members:
   :show-inheritance:

Circuit Class
~~~~~~~~~~~~~

.. autoclass:: ir.circuit.Circuit
   :members:
   :undoc-members:
   :show-inheritance:

Op Class
~~~~~~~~

.. autoclass:: ir.circuit.Op
   :members:
   :undoc-members:
   :show-inheritance:

Lowering Module
---------------

.. automodule:: qasm2.lower
   :members:
   :undoc-members:
   :show-inheritance:

Normalization Module
--------------------

.. automodule:: qasm2.normalize
   :members:
   :undoc-members:
   :show-inheritance:

Validation Module
-----------------

.. automodule:: qasm2.validate
   :members:
   :undoc-members:
   :show-inheritance:

AST Nodes Module
----------------

.. automodule:: qasm2.ast_nodes
   :members:
   :undoc-members:
   :show-inheritance:

Expression Evaluation Module
-----------------------------

.. automodule:: qasm2.expr_eval
   :members:
   :undoc-members:
   :show-inheritance:

Error Handling Module
---------------------

.. automodule:: qasm2.errors
   :members:
   :undoc-members:
   :show-inheritance:

QasmError Class
~~~~~~~~~~~~~~~

.. autoclass:: qasm2.errors.QasmError
   :members:
   :undoc-members:
   :show-inheritance:

GraphQOMB Adapter Module
------------------------

.. automodule:: graphqomb_adapter.converter
   :members:
   :undoc-members:
   :show-inheritance:

CLI Module
----------

.. automodule:: qasm2.cli
   :members:
   :undoc-members:
   :show-inheritance:
