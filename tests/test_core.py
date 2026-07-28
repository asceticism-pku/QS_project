from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from qs_project.core import (
    controlled_initialization,
    label_probabilities,
    make_circle_dataset,
    ordinary_objective,
    pack_parameters,
)


class CoreContractTests(unittest.TestCase):
    def test_circle_dataset_contract_and_hash_are_deterministic(self) -> None:
        first = make_circle_dataset(30)
        second = make_circle_dataset(30)
        self.assertEqual(first.x.shape, (4200, 2))
        self.assertEqual(first.y.shape, (4200,))
        self.assertTrue(np.array_equal(first.x, second.x))
        self.assertEqual(first.dataset_hash, second.dataset_hash)
        expected = (np.sum(first.x * first.x, axis=1) < 2 / np.pi).astype(int)
        self.assertTrue(np.array_equal(first.y, expected))

    def test_controlled_initialization_is_seeded_independently(self) -> None:
        theta_a, alpha_a = controlled_initialization(1, 4, 31)
        theta_b, alpha_b = controlled_initialization(1, 4, 31)
        self.assertTrue(np.array_equal(theta_a, theta_b))
        self.assertTrue(np.array_equal(alpha_a, alpha_b))

    def test_loss_semantics_on_fixed_basis_state(self) -> None:
        theta = np.zeros((1, 1, 3))
        alpha = np.zeros((1, 1, 2))
        x = np.asarray([[0.0, 0.0], [0.0, 0.0]])
        y = np.asarray([0, 1])
        params = pack_parameters(theta, alpha)
        probs = label_probabilities(theta, alpha, x[0], "n")
        self.assertTrue(np.allclose(probs, [1.0, 0.0], atol=1e-12))
        legacy = ordinary_objective(
            params,
            qubits=1,
            layers=1,
            x=x,
            y=y,
            entanglement="n",
            loss_id="legacy_amplitude",
        )
        squared = ordinary_objective(
            params,
            qubits=1,
            layers=1,
            x=x,
            y=y,
            entanglement="n",
            loss_id="paper_squared",
        )
        self.assertAlmostEqual(legacy, -0.5)
        self.assertAlmostEqual(squared, 0.5)


if __name__ == "__main__":
    unittest.main()
