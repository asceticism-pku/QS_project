"""Qiskit compilation and exact-probability audits for the author circuit.

The original simulator's ``U3`` convention is not Qiskit's convention.  For
author parameters ``(theta, phi, lambda)``, the equivalent Qiskit gate is
``u(theta, -phi, -lambda)`` up to a global phase.  This module keeps that
translation, the author's layer order, and compilation accounting in one
place.

Only one- and two-qubit circuits are in scope for the main project matrix.
All circuits are numeric (fully bound) before transpilation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


TARGET_BASIS_GATES = ("rz", "sx", "x", "cz")
TARGET_INSTRUCTIONS = frozenset((*TARGET_BASIS_GATES, "measure"))
OPTIMIZATION_LEVELS = (0, 3)
DEFAULT_SEED_TRANSPILER = 30
DEFAULT_PARITY_TOLERANCE = 1e-10


def _as_finite_parameter_tensor(theta_aux: ArrayLike) -> NDArray[np.float64]:
    """Validate and copy an author-format ``(qubit, layer, parameter)`` tensor."""

    try:
        values = np.asarray(theta_aux, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta_aux must contain fully bound numeric values") from exc

    if values.ndim != 3:
        raise ValueError(
            "theta_aux must have shape (qubits, layers, 3 or 6), "
            f"got {values.shape}"
        )
    if values.shape[0] not in (1, 2):
        raise ValueError("only one- and two-qubit author circuits are supported")
    if values.shape[1] < 1:
        raise ValueError("theta_aux must contain at least one layer")
    if values.shape[2] not in (3, 6):
        raise ValueError("each author layer must contain 3 or 6 parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta_aux must contain only finite values")
    return values.copy()


def _uses_cz(entanglement: bool | str) -> bool:
    if isinstance(entanglement, (bool, np.bool_)):
        return bool(entanglement)
    if not isinstance(entanglement, str):
        raise TypeError("entanglement must be a bool or a recognized string")

    normalized = entanglement.strip().lower()
    if normalized in {"y", "yes", "true", "entangled", "cz"}:
        return True
    if normalized in {"n", "no", "false", "separable", "none"}:
        return False
    raise ValueError(
        "entanglement must identify a separable or CZ circuit "
        f"(got {entanglement!r})"
    )


def append_author_u3(
    circuit: QuantumCircuit,
    qubit: int,
    theta3: ArrayLike,
) -> None:
    """Append one gate using the original simulator's ``U3`` convention."""

    angles = np.asarray(theta3, dtype=float)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("theta3 must contain three finite, bound angles")
    circuit.u(
        float(angles[0]),
        -float(angles[1]),
        -float(angles[2]),
        qubit,
    )


def build_logical_circuit(
    theta_aux: ArrayLike,
    entanglement: bool | str = False,
    *,
    include_measurements: bool = False,
) -> QuantumCircuit:
    """Build a bound Qiskit circuit in exactly the author's layer order.

    A three-parameter layer applies one author ``U3`` to each qubit.  A
    six-parameter layer applies the first and then the second author ``U3`` to
    each qubit, matching the author's ``_double_qcircuit_*`` functions.
    For a two-qubit CZ circuit, a CZ is inserted after every non-final layer.
    """

    values = _as_finite_parameter_tensor(theta_aux)
    n_qubits, n_layers, parameters_per_layer = values.shape
    use_cz = n_qubits == 2 and _uses_cz(entanglement)
    n_clbits = n_qubits if include_measurements else 0
    logical = QuantumCircuit(n_qubits, n_clbits)

    for layer in range(n_layers):
        for qubit in range(n_qubits):
            append_author_u3(logical, qubit, values[qubit, layer, :3])
            if parameters_per_layer == 6:
                append_author_u3(logical, qubit, values[qubit, layer, 3:])
        if use_cz and layer < n_layers - 1:
            logical.cz(0, 1)

    if include_measurements:
        logical.measure(range(n_qubits), range(n_qubits))
    return logical


def bind_sample(
    theta: ArrayLike,
    alpha: ArrayLike,
    sample: ArrayLike,
) -> NDArray[np.float64]:
    """Apply the author's ``code_coords`` rule without mutating its inputs."""

    bound = _as_finite_parameter_tensor(theta)
    alpha_values = np.asarray(alpha, dtype=float)
    sample_values = np.asarray(sample, dtype=float)

    if sample_values.ndim != 1:
        raise ValueError("each sample must be a one-dimensional coordinate vector")
    expected_alpha_shape = (
        bound.shape[0],
        bound.shape[1],
        sample_values.size,
    )
    if alpha_values.shape != expected_alpha_shape:
        raise ValueError(
            f"alpha must have shape {expected_alpha_shape}, "
            f"got {alpha_values.shape}"
        )
    if not np.all(np.isfinite(alpha_values)) or not np.all(
        np.isfinite(sample_values)
    ):
        raise ValueError("alpha and sample coordinates must be finite")

    if sample_values.size <= 3:
        if sample_values.size > bound.shape[2]:
            raise ValueError("sample dimensionality exceeds the available U3 angles")
        bound[:, :, : sample_values.size] += (
            alpha_values * sample_values[np.newaxis, np.newaxis, :]
        )
    elif sample_values.size == 4:
        if bound.shape[2] != 6:
            raise ValueError("four-dimensional author encoding requires 6 parameters")
        bound[:, :, (0, 1, 3, 4)] += (
            alpha_values * sample_values[np.newaxis, np.newaxis, :]
        )
    else:
        raise ValueError("the author encoding supports at most four coordinates")
    return bound


def _without_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    if "measure" not in circuit.count_ops():
        return circuit.copy()

    measurement_free = circuit.copy_empty_like()
    measurement_seen = False
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            measurement_seen = True
            continue
        if measurement_seen:
            raise ValueError(
                "only a final measurement layer is supported by the compile audit"
            )
        measurement_free.append(
            instruction.operation,
            instruction.qubits,
            instruction.clbits,
        )
    return measurement_free


def bitstring_probabilities(circuit: QuantumCircuit) -> dict[str, float]:
    """Return exact Statevector bitstring probabilities for a bound circuit."""

    if circuit.parameters:
        raise ValueError("circuit must be fully bound before exact evaluation")
    state_circuit = _without_final_measurements(circuit)
    if "measure" in state_circuit.count_ops():
        raise ValueError("mid-circuit measurements cannot be audited by Statevector")
    probabilities = Statevector.from_instruction(
        state_circuit
    ).probabilities_dict()
    return {str(bitstring): float(value) for bitstring, value in probabilities.items()}


def compile_bound_circuit(
    logical: QuantumCircuit,
    *,
    optimization_level: int,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
) -> QuantumCircuit:
    """Compile a fully bound logical circuit to RZ/SX/X/CZ/Measure."""

    if logical.parameters:
        raise ValueError("logical circuit must be fully bound before transpilation")
    if logical.num_qubits not in (1, 2):
        raise ValueError("the project compile target supports one or two qubits")
    if optimization_level not in OPTIMIZATION_LEVELS:
        raise ValueError(
            f"optimization_level must be one of {OPTIMIZATION_LEVELS}"
        )

    transpile_options: dict[str, Any] = {
        "basis_gates": list(TARGET_BASIS_GATES),
        "initial_layout": list(range(logical.num_qubits)),
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
    }
    if logical.num_qubits == 2:
        # CZ is symmetric, while the bidirectional edges make the intended
        # fully connected two-qubit synthetic target explicit to Qiskit.
        transpile_options["coupling_map"] = [[0, 1], [1, 0]]

    compiled = transpile(logical, **transpile_options)
    if compiled.parameters:
        raise AssertionError("transpilation unexpectedly produced unbound parameters")

    instruction_names = {instruction.operation.name for instruction in compiled.data}
    unsupported = instruction_names - TARGET_INSTRUCTIONS
    if unsupported:
        raise AssertionError(
            "compiled circuit contains instructions outside the target: "
            f"{sorted(unsupported)}"
        )
    return compiled


def depth_without_measurements(circuit: QuantumCircuit) -> int:
    """Return circuit depth after removing final measurement operations."""

    return int(_without_final_measurements(circuit).depth())


def target_gate_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Return RZ/SX/X/CZ counts, including explicit zero-valued entries."""

    counts = circuit.count_ops()
    return {gate: int(counts.get(gate, 0)) for gate in TARGET_BASIS_GATES}


def max_probability_error(
    expected: dict[str, float],
    actual: dict[str, float],
) -> float:
    """Compute the maximum absolute error over the union of bitstrings."""

    bitstrings = set(expected) | set(actual)
    return max(
        (abs(expected.get(key, 0.0) - actual.get(key, 0.0)) for key in bitstrings),
        default=0.0,
    )


def audit_bound_parameters(
    theta_aux: ArrayLike,
    entanglement: bool | str = False,
    *,
    sample_id: Any = 0,
    optimization_levels: Sequence[int] = OPTIMIZATION_LEVELS,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    parity_tolerance: float = DEFAULT_PARITY_TOLERANCE,
    include_measurements: bool = False,
) -> list[dict[str, Any]]:
    """Audit one fully bound parameter tensor at level 0 and/or level 3."""

    if parity_tolerance < 0:
        raise ValueError("parity_tolerance must be non-negative")
    values = _as_finite_parameter_tensor(theta_aux)
    use_cz = values.shape[0] == 2 and _uses_cz(entanglement)
    logical = build_logical_circuit(
        values,
        entanglement=use_cz,
        include_measurements=include_measurements,
    )
    logical_probabilities = bitstring_probabilities(logical)
    logical_depth = depth_without_measurements(logical)
    model_kind = (
        "1q"
        if values.shape[0] == 1
        else ("2q-cz" if use_cz else "2q-separable")
    )

    if isinstance(sample_id, np.generic):
        sample_id = sample_id.item()

    rows: list[dict[str, Any]] = []
    for level in optimization_levels:
        compiled = compile_bound_circuit(
            logical,
            optimization_level=level,
            seed_transpiler=seed_transpiler,
        )
        compiled_probabilities = bitstring_probabilities(compiled)
        error = max_probability_error(
            logical_probabilities,
            compiled_probabilities,
        )
        counts = target_gate_counts(compiled)
        depth = depth_without_measurements(compiled)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "optimization_level": int(level),
            "seed_transpiler": int(seed_transpiler),
            "n_qubits": int(values.shape[0]),
            "n_layers": int(values.shape[1]),
            "parameters_per_layer": int(values.shape[2]),
            "model_kind": model_kind,
            "uses_cz": use_cz,
            "fully_bound": not bool(compiled.parameters),
            "gate_counts": counts,
            "rz_count": counts["rz"],
            "sx_count": counts["sx"],
            "x_count": counts["x"],
            "cz_count": counts["cz"],
            "measure_count": int(compiled.count_ops().get("measure", 0)),
            "logical_depth_no_measure": logical_depth,
            "depth": depth,
            "depth_no_measure": depth,
            "logical_probabilities": logical_probabilities,
            "compiled_probabilities": compiled_probabilities,
            "max_probability_error": error,
            "max_abs_probability_error": error,
            "parity_tolerance": float(parity_tolerance),
            "parity_passed": error < parity_tolerance,
        }
        rows.append(row)
    return rows


def audit_parameter_samples(
    theta_aux_samples: Iterable[ArrayLike],
    entanglement: bool | str = False,
    *,
    sample_ids: Iterable[Any] | None = None,
    optimization_levels: Sequence[int] = OPTIMIZATION_LEVELS,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    parity_tolerance: float = DEFAULT_PARITY_TOLERANCE,
    include_measurements: bool = False,
) -> list[dict[str, Any]]:
    """Batch-audit an iterable of already bound author parameter tensors."""

    tensors = list(theta_aux_samples)
    ids = list(range(len(tensors))) if sample_ids is None else list(sample_ids)
    if len(ids) != len(tensors):
        raise ValueError("sample_ids and theta_aux_samples must have equal length")

    rows: list[dict[str, Any]] = []
    for sample_id, tensor in zip(ids, tensors, strict=True):
        rows.extend(
            audit_bound_parameters(
                tensor,
                entanglement=entanglement,
                sample_id=sample_id,
                optimization_levels=optimization_levels,
                seed_transpiler=seed_transpiler,
                parity_tolerance=parity_tolerance,
                include_measurements=include_measurements,
            )
        )
    return rows


def audit_samples(
    theta: ArrayLike,
    alpha: ArrayLike,
    samples: Iterable[ArrayLike],
    entanglement: bool | str = False,
    *,
    sample_ids: Iterable[Any] | None = None,
    optimization_levels: Sequence[int] = OPTIMIZATION_LEVELS,
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER,
    parity_tolerance: float = DEFAULT_PARITY_TOLERANCE,
    include_measurements: bool = False,
) -> list[dict[str, Any]]:
    """Bind coordinates with author semantics and return structured audit rows."""

    sample_values = list(samples)
    bound_samples = [
        bind_sample(theta, alpha, sample) for sample in sample_values
    ]
    return audit_parameter_samples(
        bound_samples,
        entanglement=entanglement,
        sample_ids=sample_ids,
        optimization_levels=optimization_levels,
        seed_transpiler=seed_transpiler,
        parity_tolerance=parity_tolerance,
        include_measurements=include_measurements,
    )
