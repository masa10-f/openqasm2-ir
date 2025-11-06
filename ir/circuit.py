"""Intermediate representation for quantum circuits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Op:
    """Quantum operation specification.

    Attributes
    ----------
    name : str
        Gate identifier such as ``"rx"`` or ``"cx"``.
    qubits : list[int]
        Indices of the qubits the gate acts on.
    params : tuple[float, ...]
        Gate parameters expressed in radians.
    """

    name: str
    qubits: list[int]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Normalize field values after initialization.

        Notes
        -----
        Coerces field values to their canonical Python types.
        """
        self.name = str(self.name)
        self.qubits = [int(qubit) for qubit in self.qubits]
        self.params = tuple(float(param) for param in self.params)


@dataclass
class Circuit:
    """Immutable circuit container for the intermediate representation.

    Attributes
    ----------
    n_qubits : int
        Total number of qubits in the circuit.
    ops : list[Op]
        Sequence of quantum operations executed in order.
    meas_map : list[tuple[int, int]] | None
        Optional mapping between measured qubits and classical bits.
    """

    n_qubits: int
    ops: list[Op]
    meas_map: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        """Normalize field values after initialization.

        Notes
        -----
        Ensures the dataclass contains independent container instances.
        """
        self.n_qubits = int(self.n_qubits)
        self.ops = list(self.ops)
        if self.meas_map is not None:
            self.meas_map = [(int(qubit), int(cbit)) for qubit, cbit in self.meas_map]

    def to_dict(self) -> dict[str, Any]:
        """Create a JSON-serializable dictionary representation.

        Returns
        -------
        dict[str, Any]
            Mapping suitable for JSON serialization.
        """
        meas_map: list[list[int]] | None = None
        if self.meas_map is not None:
            meas_map = [[qubit, cbit] for qubit, cbit in self.meas_map]

        return {
            "n_qubits": self.n_qubits,
            "ops": [
                {
                    "name": op.name,
                    "qubits": op.qubits,
                    "params": list(op.params),
                }
                for op in self.ops
            ],
            "meas_map": meas_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Circuit:
        """Instantiate a circuit from a dictionary description.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the circuit description.

        Returns
        -------
        Circuit
            Circuit instance constructed from the supplied dictionary.
        """
        ops_data = data.get("ops", [])
        ops = [
            Op(
                name=op_data["name"],
                qubits=list(op_data["qubits"]),
                params=tuple(op_data.get("params") or ()),
            )
            for op_data in ops_data
        ]

        meas_map_data = data.get("meas_map")
        meas_map = None
        if meas_map_data is not None:
            meas_map = [(pair[0], pair[1]) for pair in meas_map_data]

        return cls(
            n_qubits=data["n_qubits"],
            ops=ops,
            meas_map=meas_map,
        )

    def to_json(self, file_path: str) -> None:
        """Persist the circuit to a JSON file.

        Parameters
        ----------
        file_path : str
            Destination path of the JSON file.
        """
        path = Path(file_path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, file_path: str) -> Circuit:
        """Load a circuit description from a JSON file.

        Parameters
        ----------
        file_path : str
            Source path of the JSON file.

        Returns
        -------
        Circuit
            Circuit instance reconstructed from the JSON file.
        """
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
