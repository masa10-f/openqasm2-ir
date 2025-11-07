"""Demo script for converting QASMBench OpenQASM circuits into GraphQOMB circuits.

The script showcases how to leverage `openqasm2-ir`'s converter utilities together with helper
functions in this directory to analyze a handful of representative benchmarks.

Set QASMBENCH_PATH environment variable to specify the location of QASMBench repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from graphqomb_adapter.converter import qasm_file_to_graphqomb  # type: ignore[import-untyped]

import utils


def main() -> None:
    """Convert demo QASMBench circuits and display basic statistics."""
    openqasm2_ir_root = Path(__file__).resolve().parents[2]
    gates_yaml_path = openqasm2_ir_root / "gates" / "gates.yaml"

    # Require QASMBench path to be specified via environment variable
    qasmbench_env = os.environ.get("QASMBENCH_PATH")
    if not qasmbench_env:
        print("[ERROR] QASMBENCH_PATH environment variable is not set.")
        print("Please set it to the location of your QASMBench repository:")
        print("  export QASMBENCH_PATH=/path/to/QASMBench")
        print("  python demo_converter.py")
        return

    qasmbench_dir = Path(qasmbench_env) / "small"
    circuits: tuple[str, ...] = (
        "deutsch_n2",
        "fredkin_n3",
        "bell_n4",
        "grover_n2",
        "cat_state_n4",
        "iswap_n2",
    )

    print("QASMBench → GraphQOMB Conversion Demo")
    print("====================================")
    print(f"Circuits directory: {qasmbench_dir}")
    print(f"Gate mappings file: {gates_yaml_path}")

    if not qasmbench_dir.exists():
        print(f"[ERROR] Missing QASMBench directory: {qasmbench_dir}")
        return
    if not gates_yaml_path.exists():
        print(f"[ERROR] Missing gate mappings YAML: {gates_yaml_path}")
        return

    for circuit_name in circuits:
        qasm_path = qasmbench_dir / circuit_name / f"{circuit_name}.qasm"
        print(f"\n=== {circuit_name} ===")

        if not qasm_path.exists():
            print(f"[SKIP] Unable to locate QASM file: {qasm_path}")
            continue

        try:
            circuit = qasm_file_to_graphqomb(str(qasm_path), str(gates_yaml_path))
        except Exception as exc:  # noqa: BLE001 - user-facing demo script
            print(f"[ERROR] Failed to convert '{circuit_name}': {exc}")
            continue

        stats = utils.analyze_circuit(circuit)
        utils.print_circuit_stats(circuit_name, stats)


if __name__ == "__main__":
    main()
