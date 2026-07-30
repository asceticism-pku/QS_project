from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from circuitery import circuit as author_circuit  # noqa: E402
from qs_project.compile_audit import (  # noqa: E402
    TARGET_INSTRUCTIONS,
    audit_bound_parameters,
    audit_samples,
    bind_sample,
    bitstring_probabilities,
    build_logical_circuit,
    compile_bound_circuit,
    depth_without_measurements,
)


def author_probabilities(theta_aux: np.ndarray, entanglement: str) -> np.ndarray:
    state = np.asarray(author_circuit(theta_aux, entanglement).psi, dtype=complex)
    return np.abs(state) ** 2


@pytest.mark.parametrize(
    ("shape", "entanglement"),
    [
        ((1, 4, 3), "n"),
        ((2, 2, 3), "n"),
        ((2, 2, 3), "y"),
        ((1, 2, 6), "n"),
        ((2, 2, 6), "y"),
    ],
)
def test_qiskit_logical_circuit_matches_author_simulator(
    shape: tuple[int, int, int],
    entanglement: str,
) -> None:
    theta_aux = np.random.default_rng(2026).normal(size=shape)
    logical = build_logical_circuit(theta_aux, entanglement)
    qiskit_probabilities = np.asarray(
        [
            bitstring_probabilities(logical)[
                format(index, f"0{shape[0]}b")
            ]
            for index in range(2 ** shape[0])
        ]
    )

    np.testing.assert_allclose(
        qiskit_probabilities,
        author_probabilities(theta_aux, entanglement),
        rtol=0.0,
        atol=1e-14,
    )


def test_logical_gate_order_matches_circuitery_variants() -> None:
    parameters = np.zeros((2, 2, 3))

    one_qubit = build_logical_circuit(parameters[:1], False)
    separable = build_logical_circuit(parameters, "separable")
    entangled = build_logical_circuit(parameters, "cz")

    assert [item.operation.name for item in one_qubit.data] == ["u", "u"]
    assert [item.operation.name for item in separable.data] == [
        "u",
        "u",
        "u",
        "u",
    ]
    assert [item.operation.name for item in entangled.data] == [
        "u",
        "u",
        "cz",
        "u",
        "u",
    ]


@pytest.mark.parametrize("optimization_level", [0, 3])
@pytest.mark.parametrize("entanglement", ["separable", "cz"])
def test_bound_compilation_uses_only_target_and_preserves_probabilities(
    optimization_level: int,
    entanglement: str,
) -> None:
    theta_aux = np.random.default_rng(30).uniform(
        -np.pi,
        np.pi,
        size=(2, 3, 3),
    )
    logical = build_logical_circuit(
        theta_aux,
        entanglement,
        include_measurements=True,
    )
    compiled = compile_bound_circuit(
        logical,
        optimization_level=optimization_level,
    )

    assert not compiled.parameters
    assert {item.operation.name for item in compiled.data} <= TARGET_INSTRUCTIONS
    assert compiled.count_ops().get("measure", 0) == 2
    assert depth_without_measurements(compiled) < compiled.depth()

    logical_probabilities = bitstring_probabilities(logical)
    compiled_probabilities = bitstring_probabilities(compiled)
    assert max(
        abs(logical_probabilities[key] - compiled_probabilities[key])
        for key in logical_probabilities
    ) < 1e-10


def test_audit_samples_returns_json_serializable_structured_rows() -> None:
    rng = np.random.default_rng(31)
    theta = rng.normal(size=(2, 2, 3))
    alpha = rng.normal(size=(2, 2, 2))
    samples = np.asarray([[-0.4, 0.1], [0.25, 0.8], [0.0, -0.3]])

    rows = audit_samples(
        theta,
        alpha,
        samples,
        entanglement="cz",
        sample_ids=[7, 11, 19],
    )

    assert len(rows) == len(samples) * 2
    assert {(row["sample_id"], row["optimization_level"]) for row in rows} == {
        (sample_id, level)
        for sample_id in (7, 11, 19)
        for level in (0, 3)
    }
    for row in rows:
        assert row["model_kind"] == "2q-cz"
        assert row["fully_bound"] is True
        assert set(row["gate_counts"]) == {"rz", "sx", "x", "cz"}
        assert row["depth"] == row["depth_no_measure"]
        assert row["parity_passed"] is True
        assert row["max_probability_error"] < 1e-10
    json.dumps(rows)


def test_bind_sample_matches_author_coordinate_encoding_without_mutation() -> None:
    theta = np.arange(12.0).reshape(2, 2, 3)
    alpha = np.full((2, 2, 2), 2.0)
    theta_before = theta.copy()

    bound = bind_sample(theta, alpha, np.asarray([0.25, -0.5]))

    np.testing.assert_array_equal(theta, theta_before)
    np.testing.assert_allclose(bound[:, :, 0], theta[:, :, 0] + 0.5)
    np.testing.assert_allclose(bound[:, :, 1], theta[:, :, 1] - 1.0)
    np.testing.assert_allclose(bound[:, :, 2], theta[:, :, 2])


def test_unbound_and_invalid_compile_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="optimization_level"):
        compile_bound_circuit(
            build_logical_circuit(np.zeros((1, 1, 3))),
            optimization_level=2,
        )

    with pytest.raises(ValueError, match="equal length"):
        audit_samples(
            np.zeros((1, 1, 3)),
            np.zeros((1, 1, 2)),
            [[0.0, 0.0]],
            sample_ids=[],
        )


def test_audit_rows_report_measurement_free_depth() -> None:
    rows = audit_bound_parameters(
        np.random.default_rng(34).normal(size=(1, 4, 3)),
        include_measurements=True,
    )

    assert {row["optimization_level"] for row in rows} == {0, 3}
    assert all(row["measure_count"] == 1 for row in rows)
    assert all(row["depth"] > 0 for row in rows)
    assert all(row["parity_passed"] for row in rows)
