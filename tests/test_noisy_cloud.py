from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from circuitery import circuit  # noqa: E402
from qs_project.noisy_cloud import (  # noqa: E402
    DepolarizingNoise,
    author_u3,
    load_appendix_model,
    measurement_probabilities,
    scores_from_measurements,
    simulate_density_matrix,
)


def test_author_u3_and_noiseless_density_match_legacy_simulator() -> None:
    rng = np.random.default_rng(11)
    for qubits in (1, 2, 4):
        theta_aux = rng.normal(size=(qubits, 3, 3))
        entanglement = "n" if qubits == 1 else "y"
        legacy = np.asarray(circuit(theta_aux, entanglement).psi)
        rho = simulate_density_matrix(
            theta_aux,
            entanglement,
            DepolarizingNoise(0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            rho,
            np.outer(legacy, np.conj(legacy)),
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.conj(author_u3([0.2, 0.3, 0.4]).T)
        @ author_u3([0.2, 0.3, 0.4]),
        np.eye(2),
        atol=1e-12,
    )


def test_measurement_probabilities_recover_xyz_eigenstates() -> None:
    plus = np.asarray([1.0, 1.0]) / np.sqrt(2)
    plus_y = np.asarray([1.0, 1.0j]) / np.sqrt(2)
    rho_x = np.outer(plus, np.conj(plus))
    rho_y = np.outer(plus_y, np.conj(plus_y))
    np.testing.assert_allclose(
        measurement_probabilities(rho_x, "x"), [1, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        measurement_probabilities(rho_y, "y"), [1, 0], atol=1e-12
    )


def test_appendix_weighted_model_loads_and_scores_z_measurement() -> None:
    model = load_appendix_model(
        chi="weighted_fidelity_chi",
        problem="crown",
        qubits=1,
        layers=10,
        entanglement="n",
    )
    assert model.weights is not None
    assert model.theta.shape == (1, 10, 3)
    assert model.alpha.shape == (1, 10, 2)
    scores = scores_from_measurements(
        model,
        {
            "x": np.asarray([0.5, 0.5]),
            "y": np.asarray([0.5, 0.5]),
            "z": np.asarray([1.0, 0.0]),
        },
    )
    np.testing.assert_allclose(scores, [model.weights[0, 0], 0.0], atol=1e-12)
