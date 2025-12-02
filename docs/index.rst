qasm2graphqomb Documentation
============================

**qasm2graphqomb** is a Python library for parsing OpenQASM 2.0 source code and converting it to a custom intermediate representation (IR). It supports quantum circuit manipulation, validation, and integration with GraphQOMB.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   optimizer
   api
   modules

Features
--------

* **OpenQASM 2.0 Parser**: Full OpenQASM 2.0 syntax support
* **Intermediate Representation (IR)**: Efficient representation and manipulation of quantum circuits
* **QASM Optimizer**: Circuit optimization using pytket with configurable optimization levels
* **GraphQOMB Integration**: Seamless conversion to GraphQOMB
* **Type Safety**: Reliable development with static type checking
* **CLI Tools**: Easy command-line usage

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install qasm2graphqomb

Basic Usage
~~~~~~~~~~~

Parse an OpenQASM 2.0 file and convert it to intermediate representation:

.. code-block:: python

   from qasm2.parser import parse_qasm_file
   from qasm2.lower import lower_program

   # Parse OpenQASM file
   ast = parse_qasm_file("circuit.qasm")

   # Convert to IR
   circuit = lower_program(ast)

   # Get circuit information
   print(f"Number of qubits: {circuit.n_qubits}")
   print(f"Number of operations: {len(circuit.ops)}")

Command Line Tool
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Convert OpenQASM file to JSON IR
   qasm-import input.qasm -o output.json

Indices and Search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
