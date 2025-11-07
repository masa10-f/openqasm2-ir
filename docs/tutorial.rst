Tutorial
========

This tutorial provides step-by-step guidance on using qasm2graphqomb, from basics to advanced usage.

Basic Workflow
--------------

1. Parsing OpenQASM 2.0
~~~~~~~~~~~~~~~~~~~~~~~

Parse OpenQASM 2.0 source code to obtain an AST.

.. code-block:: python

   from qasm2.parser import parse_qasm, parse_qasm_file

   # Parse from string
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

   # Parse from file
   ast = parse_qasm_file("circuit.qasm")

   # Inspect AST contents
   print(f"OpenQASM version: {ast.version}")
   print(f"Quantum registers: {ast.qregs}")
   print(f"Classical registers: {ast.cregs}")
   print(f"Gate definitions: {list(ast.gate_defs.keys())}")
   print(f"Operations: {len(ast.body)}")

2. Converting to IR
~~~~~~~~~~~~~~~~~~~

Convert the AST to an efficient intermediate representation (IR).

.. code-block:: python

   from qasm2.lower import lower_program

   # Convert AST to IR
   circuit = lower_program(ast)

   # Get circuit information
   print(f"Number of qubits: {circuit.n_qubits}")
   print(f"Number of operations: {len(circuit.ops)}")

   # Inspect operation details
   for i, op in enumerate(circuit.ops):
       print(f"Op {i}: {op.name} on qubits {op.qubits} with params {op.params}")

3. Saving and Loading Circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can save and load IR in JSON format.

.. code-block:: python

   from ir.circuit import Circuit

   # Save as JSON
   circuit.to_json("circuit.json")

   # Load from JSON
   loaded_circuit = Circuit.from_json("circuit.json")

   # Can also work with dictionaries
   circuit_dict = circuit.to_dict()
   restored_circuit = Circuit.from_dict(circuit_dict)

Practical Examples
------------------

Creating a Bell State
~~~~~~~~~~~~~~~~~~~~~

Create a circuit that generates a two-qubit Bell state.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.lower import lower_program

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

   # Parse and convert
   ast = parse_qasm(bell_state_qasm)
   circuit = lower_program(ast)

   print("Bell State Circuit:")
   print(f"  Qubits: {circuit.n_qubits}")
   for op in circuit.ops:
       print(f"  {op.name} on qubits {op.qubits}")

Quantum Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Defining Custom Gates
~~~~~~~~~~~~~~~~~~~~~

Define and use custom gates.

.. code-block:: python

   custom_gate_qasm = """
   OPENQASM 2.0;
   include "qelib1.inc";

   // Define custom gate
   gate my_gate(theta, phi) q {
       rx(theta) q;
       ry(phi) q;
   }

   qreg q[2];

   // Use custom gate
   my_gate(pi/4, pi/2) q[0];
   my_gate(pi/3, pi/6) q[1];
   """

   ast = parse_qasm(custom_gate_qasm)
   print(f"Defined gates: {list(ast.gate_defs.keys())}")

   circuit = lower_program(ast)
   for op in circuit.ops:
       print(f"{op.name}: qubits={op.qubits}, params={op.params}")

Converting to GraphQOMB
-----------------------

Convert to GraphQOMB format.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.lower import lower_program
   from graphqomb_adapter.converter import circuit_to_graphqomb

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

   # Convert to GraphQOMB format
   graphqomb_circuit = circuit_to_graphqomb(circuit)

   print(f"GraphQOMB circuit created with {graphqomb_circuit.nqubits} qubits")

Using CLI Tools
---------------

You can also perform conversions directly from the command line.

.. code-block:: bash

   # Convert OpenQASM file to JSON IR
   qasm-import input.qasm -o output.json

   # Output to stdout
   qasm-import input.qasm

Error Handling
--------------

Handle parsing and validation errors appropriately.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.errors import QasmError

   invalid_qasm = """
   OPENQASM 2.0;
   qreg q[2];
   unknown_gate q[0];  // Undefined gate
   """

   try:
       ast = parse_qasm(invalid_qasm)
   except QasmError as e:
       print(f"Error {e.code} at line {e.line}, column {e.column}:")
       print(f"  {e.message}")

Advanced Usage
--------------

Circuit Validation
~~~~~~~~~~~~~~~~~~

Perform normalization and validation.

.. code-block:: python

   from qasm2.parser import parse_qasm
   from qasm2.normalize import normalize_program
   from qasm2.validate import validate_program

   ast = parse_qasm(qasm_code)

   # Normalize
   normalized_ast = normalize_program(ast)

   # Validate
   validate_program(normalized_ast)

Circuit Manipulation
~~~~~~~~~~~~~~~~~~~~

Manipulate circuits in IR format.

.. code-block:: python

   from ir.circuit import Circuit, Op
   import math

   # Create a new circuit
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

   # Add operations
   new_ops = circuit.ops + [Op(name="h", qubits=[1], params=())]
   modified_circuit = Circuit(
       n_qubits=circuit.n_qubits,
       ops=new_ops,
       meas_map=circuit.meas_map
   )

Summary
-------

This tutorial covered:

1. Parsing and validating OpenQASM 2.0
2. Converting to Intermediate Representation (IR)
3. Saving and loading circuits
4. Converting to GraphQOMB
5. Using CLI tools
6. Error handling and advanced operations

For more detailed information, see the :doc:`api` reference.
