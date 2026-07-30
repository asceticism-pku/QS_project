from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from qs_project.shots import (  # noqa: E402
    clopper_pearson_interval,
    evaluate_adaptive_shots,
    evaluate_fixed_shots,
    generate_bernoulli_streams,
    run_shot_campaign,
    stratified_eval_indices,
)


def test_stratified_eval_indices_selects_500_per_class_without_replacement() -> None:
    labels = np.asarray([0] * 700 + [1] * 800)

    first = stratified_eval_indices(labels)
    second = stratified_eval_indices(labels)
    different_seed = stratified_eval_indices(labels, seed=2027)

    assert first.shape == (1000,)
    assert first.dtype == np.int64
    assert np.all(first[:-1] < first[1:])
    assert np.unique(first).size == first.size
    assert np.count_nonzero(labels[first] == 0) == 500
    assert np.count_nonzero(labels[first] == 1) == 500
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)


def test_stratified_eval_indices_rejects_an_undersized_class() -> None:
    with pytest.raises(ValueError, match="class 1 has 499"):
        stratified_eval_indices([0] * 500 + [1] * 499)


def test_clopper_pearson_uses_contract_tail_probability_and_edges() -> None:
    q = 0.05 / 6
    lower, upper = clopper_pearson_interval(
        np.asarray([0, 64, 128]),
        128,
    )

    assert lower[0] == 0.0
    assert upper[2] == 1.0
    assert upper[0] == pytest.approx(beta.ppf(1 - q, 1, 128))
    assert lower[1] == pytest.approx(beta.ppf(q, 64, 65))
    assert upper[1] == pytest.approx(beta.ppf(1 - q, 65, 64))
    assert lower[2] == pytest.approx(beta.ppf(q, 128, 1))


def test_fixed_prefix_tie_is_class_zero_and_metrics_are_complete() -> None:
    streams = np.asarray(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    result = evaluate_fixed_shots(
        streams,
        exact_p0=[0.25, 0.75],
        labels=[0, 1],
        shots=4,
        report_levels=(2, 4, 6),
        include_pointwise=True,
    )

    # The first prefix ties 2:2 and therefore predicts class 0.
    np.testing.assert_array_equal(
        result["pointwise"]["prediction"],
        np.asarray([0, 1]),
    )
    np.testing.assert_array_equal(
        result["pointwise"]["count0"],
        np.asarray([2, 1]),
    )
    assert result["accuracy"] == 1.0
    assert result["exact_label_flip_rate"] == 1.0
    assert result["mean_shots"] == 4.0
    assert result["median_shots"] == 4.0
    assert result["p90_shots"] == 4.0
    assert result["p95_shots"] == 4.0
    assert result["stop_fractions"] == {2: 0.0, 4: 1.0, 6: 0.0}
    assert result["cap_hit"] == 0
    assert result["total_shots"] == 8


def _deterministic_adaptive_streams() -> np.ndarray:
    streams = np.zeros((4, 2048), dtype=np.uint8)
    streams[0, :] = 1
    streams[1, :] = 0

    # This point is unresolved at 128 (64:64), then clearly class 0 at 512.
    streams[2, :64] = 1
    streams[2, 128:512] = 1

    # Every cumulative look is exactly tied, so the final decision is the
    # class-0 tie rule and the point is a cap hit.
    streams[3, 0::2] = 1
    return streams


def test_adaptive_stopping_levels_cap_and_summary_metrics() -> None:
    result = evaluate_adaptive_shots(
        _deterministic_adaptive_streams(),
        exact_p0=[0.9, 0.1, 0.8, 0.4],
        labels=[0, 1, 0, 0],
        include_pointwise=True,
    )

    pointwise = result["pointwise"]
    np.testing.assert_array_equal(
        pointwise["prediction"],
        np.asarray([0, 1, 0, 0]),
    )
    np.testing.assert_array_equal(
        pointwise["shots"],
        np.asarray([128, 128, 512, 2048]),
    )
    np.testing.assert_array_equal(
        pointwise["count0"],
        np.asarray([128, 0, 448, 1024]),
    )
    np.testing.assert_array_equal(
        pointwise["cap_hit"],
        np.asarray([False, False, False, True]),
    )
    assert result["accuracy"] == 1.0
    assert result["exact_label_flip_rate"] == 0.25
    assert result["mean_shots"] == 704.0
    assert result["median_shots"] == 320.0
    assert result["p90_shots"] == pytest.approx(1587.2)
    assert result["p95_shots"] == pytest.approx(1817.6)
    assert result["stop_fractions"] == {128: 0.5, 512: 0.25, 2048: 0.25}
    assert result["cap_hit"] == 1
    assert result["cap_hit_fraction"] == 0.25
    assert result["total_shots"] == 2816
    assert result["cp_tail_probability"] == pytest.approx(0.05 / 6)


def test_adaptive_stopping_does_not_read_exact_probabilities() -> None:
    streams = _deterministic_adaptive_streams()
    labels = [0, 1, 0, 0]

    first = evaluate_adaptive_shots(
        streams,
        exact_p0=[0.9, 0.1, 0.8, 0.4],
        labels=labels,
        include_pointwise=True,
    )
    second = evaluate_adaptive_shots(
        streams,
        exact_p0=[0.1, 0.9, 0.2, 0.6],
        labels=labels,
        include_pointwise=True,
    )

    for field in ("prediction", "shots", "count0", "cap_hit", "cp_lower", "cp_upper"):
        np.testing.assert_array_equal(
            first["pointwise"][field],
            second["pointwise"][field],
        )
    assert first["exact_label_flip_rate"] != second["exact_label_flip_rate"]


def test_campaign_draws_once_and_every_method_uses_the_same_prefixes() -> None:
    exact_p0 = np.asarray([0.2, 0.45, 0.55, 0.8])
    labels = np.asarray([1, 1, 0, 0])

    campaign = run_shot_campaign(
        exact_p0,
        labels,
        np.random.default_rng(81),
        include_pointwise=True,
    )
    streams = generate_bernoulli_streams(
        exact_p0,
        np.random.default_rng(81),
    )

    for shots in (128, 512, 2048):
        independently_evaluated = evaluate_fixed_shots(
            streams,
            exact_p0,
            labels,
            shots,
            include_pointwise=True,
        )
        np.testing.assert_array_equal(
            campaign["fixed"][shots]["pointwise"]["count0"],
            independently_evaluated["pointwise"]["count0"],
        )
        np.testing.assert_array_equal(
            campaign["fixed"][shots]["pointwise"]["count0"],
            np.sum(streams[:, :shots], axis=1),
        )

    independently_adaptive = evaluate_adaptive_shots(
        streams,
        exact_p0,
        labels,
        include_pointwise=True,
    )
    for field in ("prediction", "shots", "count0", "cap_hit"):
        np.testing.assert_array_equal(
            campaign["adaptive"]["pointwise"][field],
            independently_adaptive["pointwise"][field],
        )


def test_campaign_is_reproducible_for_an_explicit_campaign_rng() -> None:
    exact_p0 = np.linspace(0.1, 0.9, 20)
    labels = (exact_p0 < 0.5).astype(int)

    first = run_shot_campaign(
        exact_p0,
        labels,
        np.random.default_rng(2026),
        include_pointwise=True,
    )
    second = run_shot_campaign(
        exact_p0,
        labels,
        np.random.default_rng(2026),
        include_pointwise=True,
    )

    for shots in (128, 512, 2048):
        np.testing.assert_array_equal(
            first["fixed"][shots]["pointwise"]["prediction"],
            second["fixed"][shots]["pointwise"]["prediction"],
        )
    np.testing.assert_array_equal(
        first["adaptive"]["pointwise"]["shots"],
        second["adaptive"]["pointwise"]["shots"],
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: generate_bernoulli_streams([0.2, 1.2], np.random.default_rng(0)),
            r"\[0, 1\]",
        ),
        (
            lambda: generate_bernoulli_streams([0.2], 30),
            "explicit numpy.random.Generator",
        ),
        (
            lambda: evaluate_fixed_shots([[1, 2]], [0.5], [0], 1),
            "binary outcomes",
        ),
        (
            lambda: evaluate_adaptive_shots(
                np.zeros((1, 128)),
                [0.5],
                [0],
            ),
            "only 128 are available",
        ),
        (
            lambda: clopper_pearson_interval(129, 128),
            "between 0 and shots",
        ),
    ],
)
def test_invalid_inputs_are_rejected(call: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
