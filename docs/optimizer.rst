QASM Optimizer
==============

The ``optimizer`` module provides tools for optimizing OpenQASM 2.0 quantum circuits
using `pytket <https://docs.quantinuum.com/tket/>`_'s optimization passes.

Installation
------------

The optimizer is included as a default dependency. When you install the package,
pytket is automatically installed:

.. code-block:: bash

   pip install qasm2graphqomb

Optimization Levels
-------------------

The optimizer provides 5 optimization levels (0-4), allowing you to balance
between optimization quality and processing time:

.. list-table:: Optimization Levels
   :header-rows: 1
   :widths: 10 25 65

   * - Level
     - Enum Name
     - Description
   * - 0
     - ``MINIMAL``
     - RemoveRedundancies only. Fastest optimization that removes gate-inverse pairs and zero-angle rotations.
   * - 1
     - ``CLIFFORD``
     - CliffordSimp + RemoveRedundancies. Simplifies Clifford gates.
   * - 2
     - ``PEEPHOLE_2Q``
     - PeepholeOptimise2Q + RemoveRedundancies. **Default level** with good balance between quality and speed.
   * - 3
     - ``FULL_NO_SWAPS``
     - FullPeepholeOptimise (no swaps) + RemoveRedundancies. Comprehensive optimization without SWAP gate insertion.
   * - 4
     - ``FULL_WITH_SWAPS``
     - FullPeepholeOptimise (with swaps) + RemoveRedundancies. Most aggressive optimization that may insert SWAP gates.

Quick Start
-----------

Python API
~~~~~~~~~~

**File to file optimization:**

.. code-block:: python

   from optimizer import optimise_qasm_with_tket, OptimizationLevel

   # Default optimization (level 2)
   optimise_qasm_with_tket("input.qasm", "output.qasm")

   # Fast optimization (level 0)
   optimise_qasm_with_tket("input.qasm", "output.qasm", optimization_level=0)

   # Most aggressive (level 4)
   optimise_qasm_with_tket(
       "input.qasm",
       "output.qasm",
       optimization_level=OptimizationLevel.FULL_WITH_SWAPS
   )

**File to string optimization (useful for GraphQOMB integration):**

.. code-block:: python

   from optimizer import optimise_qasm_to_string

   optimized_qasm = optimise_qasm_to_string("input.qasm", optimization_level=2)

   # Use directly with GraphQOMB
   from qasm2.parser import parse_qasm
   ast = parse_qasm(optimized_qasm)

**String to string optimization (most flexible):**

.. code-block:: python

   from optimizer import optimise_qasm_string, OptimizationLevel

   qasm_code = """OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[2];
   h q[0];
   x q[0];
   x q[0];
   cx q[0], q[1];
   """

   # The redundant x-x pair will be removed
   optimized = optimise_qasm_string(qasm_code, optimization_level=OptimizationLevel.MINIMAL)

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

After installation, the ``tket-optimize-qasm`` command is available:

.. code-block:: bash

   # Default optimization (level 2)
   tket-optimize-qasm input.qasm output.qasm

   # Specify optimization level
   tket-optimize-qasm input.qasm output.qasm --level 0  # fastest
   tket-optimize-qasm input.qasm output.qasm --level 4  # most aggressive

   # Verbose output
   tket-optimize-qasm input.qasm output.qasm --level 2 -v

Performance Comparison
----------------------

Here's an example benchmark on a 26-qubit Ising model circuit (ising_n26.qasm):

.. list-table:: Optimization Results
   :header-rows: 1
   :widths: 50 15 15 10 10

   * - Optimization Level
     - Gates
     - Depth
     - Gate Δ
     - Time
   * - Original
     - 307
     - 16
     - —
     - —
   * - Level 0: MINIMAL
     - 178
     - 10
     - -42%
     - 0.002s
   * - Level 1: CLIFFORD
     - 152
     - 9
     - -51%
     - 0.029s
   * - Level 2: PEEPHOLE_2Q (default)
     - 140
     - 9
     - -54%
     - 0.054s
   * - Level 3: FULL_NO_SWAPS
     - 140
     - 9
     - -54%
     - 0.288s
   * - Level 4: FULL_WITH_SWAPS
     - 140
     - 9
     - -54%
     - 0.291s

As shown, level 2 (PEEPHOLE_2Q) provides an excellent balance between optimization
quality and processing time for most use cases.

API Reference
-------------

OptimizationLevel
~~~~~~~~~~~~~~~~~

.. py:class:: optimizer.OptimizationLevel

   Enumeration of optimization levels.

   .. py:attribute:: MINIMAL
      :value: 0

      RemoveRedundancies only (fastest).

   .. py:attribute:: CLIFFORD
      :value: 1

      CliffordSimp + RemoveRedundancies.

   .. py:attribute:: PEEPHOLE_2Q
      :value: 2

      PeepholeOptimise2Q + RemoveRedundancies (default).

   .. py:attribute:: FULL_NO_SWAPS
      :value: 3

      FullPeepholeOptimise without SWAP insertion.

   .. py:attribute:: FULL_WITH_SWAPS
      :value: 4

      FullPeepholeOptimise with SWAP insertion (most aggressive).

   .. py:property:: description
      :type: str

      Human-readable description of the optimization level.

Functions
~~~~~~~~~

.. py:function:: optimizer.optimise_qasm_with_tket(in_qasm_path, out_qasm_path, *, optimization_level=OptimizationLevel.PEEPHOLE_2Q, allow_swaps=None)

   Optimize an OpenQASM file and write the result to another file.

   :param in_qasm_path: Path to the input QASM file.
   :type in_qasm_path: str or Path
   :param out_qasm_path: Path to the output (optimized) QASM file.
   :type out_qasm_path: str or Path
   :param optimization_level: The optimization level (0-4). Default is 2.
   :type optimization_level: int or OptimizationLevel
   :param allow_swaps: Deprecated. Use ``optimization_level`` instead.
   :type allow_swaps: bool, optional
   :raises ImportError: If pytket is not installed.
   :raises FileNotFoundError: If the input file does not exist.
   :raises ValueError: If the QASM file cannot be parsed.

.. py:function:: optimizer.optimise_qasm_to_string(in_qasm_path, *, optimization_level=OptimizationLevel.PEEPHOLE_2Q, allow_swaps=None)

   Optimize an OpenQASM file and return the result as a string.

   :param in_qasm_path: Path to the input QASM file.
   :type in_qasm_path: str or Path
   :param optimization_level: The optimization level (0-4). Default is 2.
   :type optimization_level: int or OptimizationLevel
   :param allow_swaps: Deprecated. Use ``optimization_level`` instead.
   :type allow_swaps: bool, optional
   :returns: The optimized QASM code.
   :rtype: str
   :raises ImportError: If pytket is not installed.
   :raises FileNotFoundError: If the input file does not exist.
   :raises ValueError: If the QASM file cannot be parsed.

.. py:function:: optimizer.optimise_qasm_string(qasm_string, *, optimization_level=OptimizationLevel.PEEPHOLE_2Q, allow_swaps=None)

   Optimize a QASM string and return the optimized result.

   :param qasm_string: The input QASM code as a string.
   :type qasm_string: str
   :param optimization_level: The optimization level (0-4). Default is 2.
   :type optimization_level: int or OptimizationLevel
   :param allow_swaps: Deprecated. Use ``optimization_level`` instead.
   :type allow_swaps: bool, optional
   :returns: The optimized QASM code.
   :rtype: str
   :raises ImportError: If pytket is not installed.
   :raises ValueError: If the QASM string cannot be parsed.

Integration with GraphQOMB
--------------------------

The optimizer integrates seamlessly with the GraphQOMB conversion pipeline:

.. code-block:: python

   from optimizer import optimise_qasm_to_string, OptimizationLevel
   from qasm2.parser import parse_qasm
   from qasm2.lower import lower_program

   # Optimize QASM first
   optimized_qasm = optimise_qasm_to_string(
       "circuit.qasm",
       optimization_level=OptimizationLevel.PEEPHOLE_2Q
   )

   # Parse the optimized QASM
   ast = parse_qasm(optimized_qasm)

   # Lower to IR
   circuit = lower_program(ast)

   # Continue with GraphQOMB conversion...

References
----------

- `pytket Documentation <https://docs.quantinuum.com/tket/>`_
- `pytket Compilation Guide <https://docs.quantinuum.com/tket/user-guide/manual/manual_compiler.html>`_
- `pytket.qasm Module <https://docs.quantinuum.com/tket/api-docs/qasm.html>`_
