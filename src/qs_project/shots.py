"""Finite-shot and confidence-sequential evaluation for frozen classifiers.

The event represented by a one in a shot stream is measurement outcome
``0``.  Consequently, ``count0`` is binomial with parameter ``exact_p0`` and
ties are classified as class 0.

One campaign must call :func:`generate_bernoulli_streams` exactly once.  The
fixed-shot baselines and the adaptive policy are then evaluated from prefixes
of that same array.  :func:`run_shot_campaign` enforces this construction.
Exact probabilities are used only to generate the stream and to score
``exact_label_flip_rate``; adaptive stopping reads counts alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta


DEFAULT_FIXED_SHOTS = (128, 512, 2048)
DEFAULT_ADAPTIVE_CHECKPOINTS = (128, 512, 2048)
DEFAULT_ALPHA = 0.05
DEFAULT_EVAL_SEED = 2026
DEFAULT_POINTS_PER_CLASS = 500


def _as_probabilities(exact_p0: ArrayLike) -> NDArray[np.float64]:
    try:
        probabilities = np.asarray(exact_p0, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("exact_p0 must be a one-dimensional numeric array") from exc

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("exact_p0 must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("exact_p0 must contain only finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("exact_p0 values must lie in [0, 1]")
    return probabilities.copy()


def _as_binary_labels(
    labels: ArrayLike,
    *,
    expected_size: int | None = None,
) -> NDArray[np.int8]:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if expected_size is not None and values.size != expected_size:
        raise ValueError(
            "exact_p0 and labels must have equal length "
            f"({expected_size} != {values.size})"
        )
    try:
        is_binary = np.asarray((values == 0) | (values == 1), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must contain only class 0 or class 1") from exc
    if not np.all(is_binary):
        raise ValueError("labels must contain only class 0 or class 1")
    return values.astype(np.int8, copy=True)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _shot_levels(
    levels: Sequence[int],
    *,
    name: str,
    max_available: int | None = None,
) -> tuple[int, ...]:
    normalized = tuple(
        _positive_integer(value, name=f"{name} entry") for value in levels
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one shot count")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    if max_available is not None and normalized[-1] > max_available:
        raise ValueError(
            f"{name} requires {normalized[-1]} shots, "
            f"but only {max_available} are available"
        )
    return normalized


def _as_streams(
    streams: ArrayLike,
    *,
    expected_points: int,
) -> NDArray[np.uint8]:
    values = np.asarray(streams)
    if values.ndim != 2:
        raise ValueError("streams must have shape (points, shots)")
    if values.shape[0] != expected_points:
        raise ValueError(
            "streams and labels must contain the same number of points "
            f"({values.shape[0]} != {expected_points})"
        )
    if values.shape[1] == 0:
        raise ValueError("streams must contain at least one shot per point")
    try:
        is_binary = np.asarray((values == 0) | (values == 1), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError("streams must contain only binary outcomes") from exc
    if not np.all(is_binary):
        raise ValueError("streams must contain only binary outcomes")
    return values.astype(np.uint8, copy=False)


def exact_predictions(exact_p0: ArrayLike) -> NDArray[np.int8]:
    """Return exact-probability labels, assigning an exact tie to class 0."""

    probabilities = _as_probabilities(exact_p0)
    return np.where(probabilities >= 0.5, 0, 1).astype(np.int8)


def stratified_eval_indices(
    labels: ArrayLike,
    *,
    per_class: int = DEFAULT_POINTS_PER_CLASS,
    seed: int = DEFAULT_EVAL_SEED,
) -> NDArray[np.int64]:
    """Select ``per_class`` indices from each binary class without replacement.

    The selected indices are sorted before return, so downstream evaluation
    preserves original dataset order.  Selection itself uses a local NumPy
    generator and never changes global RNG state.
    """

    binary_labels = _as_binary_labels(labels)
    sample_size = _positive_integer(per_class, name="per_class")
    rng = np.random.default_rng(seed)
    selected: list[NDArray[np.int64]] = []

    for class_id in (0, 1):
        candidates = np.flatnonzero(binary_labels == class_id)
        if candidates.size < sample_size:
            raise ValueError(
                f"class {class_id} has {candidates.size} points; "
                f"{sample_size} are required"
            )
        class_indices = rng.choice(
            candidates,
            size=sample_size,
            replace=False,
        )
        selected.append(np.asarray(class_indices, dtype=np.int64))

    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def generate_bernoulli_streams(
    exact_p0: ArrayLike,
    rng: np.random.Generator,
    *,
    max_shots: int = DEFAULT_ADAPTIVE_CHECKPOINTS[-1],
) -> NDArray[np.uint8]:
    """Generate one longest Bernoulli outcome-0 stream for every eval point."""

    probabilities = _as_probabilities(exact_p0)
    n_shots = _positive_integer(max_shots, name="max_shots")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    return (
        rng.random((probabilities.size, n_shots))
        < probabilities[:, np.newaxis]
    ).astype(np.uint8)


def clopper_pearson_interval(
    count0: ArrayLike,
    shots: int,
    *,
    tail_probability: float = DEFAULT_ALPHA
    / (2 * len(DEFAULT_ADAPTIVE_CHECKPOINTS)),
) -> (
    tuple[float, float]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
):
    """Return an exact two-sided binomial interval for ``P(outcome=0)``.

    ``tail_probability`` is the probability in each tail.  The M4 contract
    uses ``0.05 / 6`` because there are three cumulative looks.
    """

    n_shots = _positive_integer(shots, name="shots")
    if not np.isfinite(tail_probability) or not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie strictly between 0 and 0.5")

    try:
        raw_counts = np.asarray(count0, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("count0 must contain integer counts") from exc
    if not np.all(np.isfinite(raw_counts)):
        raise ValueError("count0 must contain finite counts")
    if np.any(raw_counts != np.floor(raw_counts)):
        raise ValueError("count0 must contain integer counts")
    if np.any((raw_counts < 0) | (raw_counts > n_shots)):
        raise ValueError("count0 must lie between 0 and shots")

    scalar_input = raw_counts.ndim == 0
    counts = raw_counts.astype(np.int64, copy=False)
    lower = np.zeros(counts.shape, dtype=float)
    upper = np.ones(counts.shape, dtype=float)

    has_successes = counts > 0
    if np.any(has_successes):
        lower[has_successes] = beta.ppf(
            tail_probability,
            counts[has_successes],
            n_shots - counts[has_successes] + 1,
        )

    has_failures = counts < n_shots
    if np.any(has_failures):
        upper[has_failures] = beta.ppf(
            1.0 - tail_probability,
            counts[has_failures] + 1,
            n_shots - counts[has_failures],
        )

    if scalar_input:
        return float(lower), float(upper)
    return lower, upper


def _summarize_predictions(
    predictions: NDArray[np.int8],
    labels: NDArray[np.int8],
    exact_labels: NDArray[np.int8],
    shots_used: NDArray[np.int64],
    count0: NDArray[np.int64],
    cap_hit_mask: NDArray[np.bool_],
    *,
    report_levels: Sequence[int],
    include_pointwise: bool,
    extra_pointwise: dict[str, NDArray[Any]] | None = None,
) -> dict[str, Any]:
    correct = predictions == labels
    flips = predictions != exact_labels
    levels = tuple(int(level) for level in report_levels)
    stop_fractions = {
        level: float(np.mean(shots_used == level)) for level in levels
    }

    summary: dict[str, Any] = {
        "n_points": int(labels.size),
        "accuracy": float(np.mean(correct)),
        "exact_label_flip_rate": float(np.mean(flips)),
        "mean_shots": float(np.mean(shots_used)),
        "median_shots": float(np.median(shots_used)),
        "p90_shots": float(np.percentile(shots_used, 90)),
        "p95_shots": float(np.percentile(shots_used, 95)),
        "stop_fractions": stop_fractions,
        "cap_hit": int(np.count_nonzero(cap_hit_mask)),
        "cap_hit_fraction": float(np.mean(cap_hit_mask)),
        "total_shots": int(np.sum(shots_used, dtype=np.int64)),
    }
    for level, fraction in stop_fractions.items():
        summary[f"stop_fraction_{level}"] = fraction

    if include_pointwise:
        pointwise: dict[str, NDArray[Any]] = {
            "prediction": predictions.copy(),
            "exact_label": exact_labels.copy(),
            "correct": correct.copy(),
            "exact_label_flip": flips.copy(),
            "shots": shots_used.copy(),
            "count0": count0.copy(),
            "count1": shots_used - count0,
            "cap_hit": cap_hit_mask.copy(),
        }
        if extra_pointwise is not None:
            pointwise.update(
                {key: np.asarray(value).copy() for key, value in extra_pointwise.items()}
            )
        summary["pointwise"] = pointwise
    return summary


def evaluate_fixed_shots(
    streams: ArrayLike,
    exact_p0: ArrayLike,
    labels: ArrayLike,
    shots: int,
    *,
    report_levels: Sequence[int] = DEFAULT_ADAPTIVE_CHECKPOINTS,
    include_pointwise: bool = False,
) -> dict[str, Any]:
    """Evaluate one fixed-shot baseline from a prefix of existing streams."""

    probabilities = _as_probabilities(exact_p0)
    binary_labels = _as_binary_labels(labels, expected_size=probabilities.size)
    outcomes = _as_streams(streams, expected_points=probabilities.size)
    n_shots = _positive_integer(shots, name="shots")
    if n_shots > outcomes.shape[1]:
        raise ValueError(
            f"{n_shots} fixed shots requested, but streams contain "
            f"{outcomes.shape[1]}"
        )
    levels = _shot_levels(report_levels, name="report_levels")

    count0 = np.sum(outcomes[:, :n_shots], axis=1, dtype=np.int64)
    predictions = np.where(2 * count0 >= n_shots, 0, 1).astype(np.int8)
    shots_used = np.full(probabilities.size, n_shots, dtype=np.int64)
    cap_hit_mask = np.zeros(probabilities.size, dtype=bool)
    return _summarize_predictions(
        predictions,
        binary_labels,
        exact_predictions(probabilities),
        shots_used,
        count0,
        cap_hit_mask,
        report_levels=levels,
        include_pointwise=include_pointwise,
    )


def evaluate_adaptive_shots(
    streams: ArrayLike,
    exact_p0: ArrayLike,
    labels: ArrayLike,
    *,
    checkpoints: Sequence[int] = DEFAULT_ADAPTIVE_CHECKPOINTS,
    alpha: float = DEFAULT_ALPHA,
    include_pointwise: bool = False,
) -> dict[str, Any]:
    """Evaluate the cumulative Clopper--Pearson stopping policy.

    The policy observes only prefix counts.  At the final checkpoint, points
    whose interval still crosses 0.5 are classified by counts argmax; a count
    tie is class 0 and every such unresolved point is marked ``cap_hit``.
    """

    probabilities = _as_probabilities(exact_p0)
    binary_labels = _as_binary_labels(labels, expected_size=probabilities.size)
    outcomes = _as_streams(streams, expected_points=probabilities.size)
    looks = _shot_levels(
        checkpoints,
        name="checkpoints",
        max_available=outcomes.shape[1],
    )
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    tail_probability = float(alpha) / (2 * len(looks))

    n_points = probabilities.size
    predictions = np.full(n_points, -1, dtype=np.int8)
    shots_used = np.zeros(n_points, dtype=np.int64)
    final_count0 = np.zeros(n_points, dtype=np.int64)
    final_lower = np.full(n_points, np.nan, dtype=float)
    final_upper = np.full(n_points, np.nan, dtype=float)
    cap_hit_mask = np.zeros(n_points, dtype=bool)
    active = np.ones(n_points, dtype=bool)

    for n_shots in looks:
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        counts = np.sum(
            outcomes[active_indices, :n_shots],
            axis=1,
            dtype=np.int64,
        )
        lower, upper = clopper_pearson_interval(
            counts,
            n_shots,
            tail_probability=tail_probability,
        )
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        predict_zero = lower > 0.5
        predict_one = upper < 0.5
        separated = predict_zero | predict_one

        if np.any(separated):
            stopped_indices = active_indices[separated]
            predictions[stopped_indices] = np.where(
                predict_zero[separated],
                0,
                1,
            )
            shots_used[stopped_indices] = n_shots
            final_count0[stopped_indices] = counts[separated]
            final_lower[stopped_indices] = lower[separated]
            final_upper[stopped_indices] = upper[separated]
            active[stopped_indices] = False

        if n_shots == looks[-1] and np.any(~separated):
            unresolved_indices = active_indices[~separated]
            unresolved_count0 = counts[~separated]
            predictions[unresolved_indices] = np.where(
                2 * unresolved_count0 >= n_shots,
                0,
                1,
            )
            shots_used[unresolved_indices] = n_shots
            final_count0[unresolved_indices] = unresolved_count0
            final_lower[unresolved_indices] = lower[~separated]
            final_upper[unresolved_indices] = upper[~separated]
            cap_hit_mask[unresolved_indices] = True
            active[unresolved_indices] = False

    if np.any(active) or np.any(predictions < 0):
        raise AssertionError("adaptive evaluation left points without a prediction")

    summary = _summarize_predictions(
        predictions,
        binary_labels,
        exact_predictions(probabilities),
        shots_used,
        final_count0,
        cap_hit_mask,
        report_levels=looks,
        include_pointwise=include_pointwise,
        extra_pointwise={
            "cp_lower": final_lower,
            "cp_upper": final_upper,
        },
    )
    summary["alpha"] = float(alpha)
    summary["cp_tail_probability"] = tail_probability
    summary["checkpoints"] = list(looks)
    return summary


def run_shot_campaign(
    exact_p0: ArrayLike,
    labels: ArrayLike,
    rng: np.random.Generator,
    *,
    fixed_shots: Sequence[int] = DEFAULT_FIXED_SHOTS,
    adaptive_checkpoints: Sequence[int] = DEFAULT_ADAPTIVE_CHECKPOINTS,
    alpha: float = DEFAULT_ALPHA,
    include_pointwise: bool = False,
) -> dict[str, Any]:
    """Run all fixed baselines and adaptive evaluation on one shared stream."""

    probabilities = _as_probabilities(exact_p0)
    binary_labels = _as_binary_labels(labels, expected_size=probabilities.size)
    fixed_levels = _shot_levels(fixed_shots, name="fixed_shots")
    adaptive_levels = _shot_levels(
        adaptive_checkpoints,
        name="adaptive_checkpoints",
    )
    max_shots = max(fixed_levels[-1], adaptive_levels[-1])
    streams = generate_bernoulli_streams(
        probabilities,
        rng,
        max_shots=max_shots,
    )

    fixed = {
        n_shots: evaluate_fixed_shots(
            streams,
            probabilities,
            binary_labels,
            n_shots,
            report_levels=adaptive_levels,
            include_pointwise=include_pointwise,
        )
        for n_shots in fixed_levels
    }
    adaptive = evaluate_adaptive_shots(
        streams,
        probabilities,
        binary_labels,
        checkpoints=adaptive_levels,
        alpha=alpha,
        include_pointwise=include_pointwise,
    )

    campaign: dict[str, Any] = {
        "n_points": int(probabilities.size),
        "max_shots_per_point": int(max_shots),
        "fixed_shots": list(fixed_levels),
        "adaptive_checkpoints": list(adaptive_levels),
        "alpha": float(alpha),
        "fixed": fixed,
        "adaptive": adaptive,
    }
    if include_pointwise:
        campaign["pointwise"] = {
            "exact_p0": probabilities.copy(),
            "label": binary_labels.copy(),
        }
    return campaign


def simulate_shot_campaign(
    exact_p0: ArrayLike,
    labels: ArrayLike,
    rng: np.random.Generator,
    *,
    fixed_shots: Sequence[int] = DEFAULT_FIXED_SHOTS,
    adaptive_checkpoints: Sequence[int] = DEFAULT_ADAPTIVE_CHECKPOINTS,
    alpha: float = DEFAULT_ALPHA,
    include_pointwise: bool = False,
) -> dict[str, Any]:
    """Alias with an experiment-facing name for :func:`run_shot_campaign`."""

    return run_shot_campaign(
        exact_p0,
        labels,
        rng,
        fixed_shots=fixed_shots,
        adaptive_checkpoints=adaptive_checkpoints,
        alpha=alpha,
        include_pointwise=include_pointwise,
    )


__all__ = [
    "DEFAULT_ADAPTIVE_CHECKPOINTS",
    "DEFAULT_ALPHA",
    "DEFAULT_EVAL_SEED",
    "DEFAULT_FIXED_SHOTS",
    "DEFAULT_POINTS_PER_CLASS",
    "clopper_pearson_interval",
    "evaluate_adaptive_shots",
    "evaluate_fixed_shots",
    "exact_predictions",
    "generate_bernoulli_streams",
    "run_shot_campaign",
    "simulate_shot_campaign",
    "stratified_eval_indices",
]
