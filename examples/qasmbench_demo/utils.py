"""Utilities for analyzing and displaying GraphQOMB circuit statistics."""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

from graphqomb.circuit import Circuit


class CircuitStats(TypedDict):
    """Container for circuit statistics."""

    num_qubits: int
    total_gates: int
    gate_type_counts: Counter[str]


def analyze_circuit(circuit: Circuit) -> CircuitStats:
    """Compute summary statistics for a GraphQOMB circuit.

    Parameters
    ----------
    circuit : `Circuit`
        Circuit instance to analyze.

    Returns
    -------
    `CircuitStats`
        Dictionary containing qubit count, total gate count, and gate type histogram.
    """
    instructions = circuit.instructions()
    gate_names = (instruction.__class__.__name__ for instruction in instructions)
    gate_type_counts = Counter(gate_names)

    return {
        "num_qubits": circuit.num_qubits,
        "total_gates": len(instructions),
        "gate_type_counts": gate_type_counts,
    }


def print_circuit_stats(name: str, stats: CircuitStats) -> None:
    """Pretty-print circuit statistics for console display.

    Parameters
    ----------
    name : `str`
        Human-readable circuit name.
    stats : `CircuitStats`
        Statistics produced by `analyze_circuit`.
    """
    header = f"Circuit Statistics — {name}"
    print(header)
    print("-" * len(header))
    print(f"Qubits      : {stats['num_qubits']}")
    print(f"Total gates : {stats['total_gates']}")

    gate_counts = stats["gate_type_counts"]
    print("Gate breakdown:")
    if gate_counts:
        padding = max(len(gate) for gate in gate_counts)
        for gate, count in gate_counts.most_common():
            print(f"  - {gate:<{padding}} : {count}")
    else:
        print("  (no gates)")
