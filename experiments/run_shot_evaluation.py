#!/usr/bin/env python3
"""Run the frozen-checkpoint M4 fixed/adaptive finite-shot evaluation.

This runner never calls an optimizer.  It verifies and freezes the five M1
``1q-l4-paper_squared`` checkpoints and the five M3
``l4-to-l3-pruned`` checkpoints, evaluates exact probabilities once, and then
runs 100 finite-shot campaigns per checkpoint.

Every campaign draws one ``(1000, 2048)`` Bernoulli array.  Fixed 128/512/2048
and adaptive 128 -> 512 -> 2048 are all computed from prefixes of that same
array by :mod:`qs_project.shots`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    TRAIN_COUNT,
    CircleDataset,
    canonical_json_hash,
    create_run_directory,
    environment_record,
    git_revision,
    json_dump,
    label_probabilities,
    load_checkpoint,
    make_circle_dataset,
    save_dataset,
    sha256_file,
)
from qs_project.shots import (  # noqa: E402
    DEFAULT_ADAPTIVE_CHECKPOINTS,
    DEFAULT_ALPHA,
    DEFAULT_EVAL_SEED,
    DEFAULT_FIXED_SHOTS,
    DEFAULT_POINTS_PER_CLASS,
    exact_predictions,
    run_shot_campaign,
    stratified_eval_indices,
)


EXPERIMENT_ID = "M4"
CONFIG_ID = "fixed-adaptive-shots"
EVIDENCE_LABEL = "shot-simulation"
EVAL_SEED = DEFAULT_EVAL_SEED
POINTS_PER_CLASS = DEFAULT_POINTS_PER_CLASS
CAMPAIGN_REPEATS = 100
CAMPAIGN_MASTER_SEED = 2026
RNG_NAMESPACE = 4
PROGRESS_EVERY = 10


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_code: int
    source_experiment_id: str
    source_config_id: str
    layers: int


MODEL_SPECS = (
    ModelSpec(
        model_id="l4-base",
        model_code=0,
        source_experiment_id="M1",
        source_config_id="1q-l4-paper_squared",
        layers=4,
    ),
    ModelSpec(
        model_id="l4-to-l3-pruned",
        model_code=1,
        source_experiment_id="M3",
        source_config_id="l4-to-l3-pruned",
        layers=3,
    ),
)

CAMPAIGN_METRIC_FIELDS = (
    "accuracy",
    "exact_label_flip_rate",
    "mean_shots",
    "median_shots",
    "p90_shots",
    "p95_shots",
    "stop_fraction_128",
    "stop_fraction_512",
    "stop_fraction_2048",
    "cap_hit",
    "cap_hit_fraction",
    "total_shots",
)


def exact_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty campaign CSV")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_npz_exclusive(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def evaluation_source_hashes() -> dict[str, str]:
    relative_paths = (
        Path("src/qs_project/core.py"),
        Path("src/qs_project/shots.py"),
        Path("experiments/run_shot_evaluation.py"),
    )
    return {
        str(relative): sha256_file(ROOT / relative) for relative in relative_paths
    }


def eval_indices_artifact(
    dataset: CircleDataset,
    *,
    root: Path = ROOT,
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    """Create or contract-check the shared 500-per-class M4 eval indices."""

    relative_indices = stratified_eval_indices(
        dataset.test_y,
        per_class=POINTS_PER_CLASS,
        seed=EVAL_SEED,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "M4-fixed-adaptive-shots",
        "eval_seed": EVAL_SEED,
        "sampling": "stratified-without-replacement",
        "per_class": POINTS_PER_CLASS,
        "point_count": int(relative_indices.size),
        "index_space": "test-set-relative",
        "test_relative_indices": relative_indices.tolist(),
        "dataset_global_indices": (relative_indices + TRAIN_COUNT).tolist(),
        "labels": dataset.test_y[relative_indices].astype(int).tolist(),
        "dataset_hash": dataset.dataset_hash,
    }
    payload["indices_content_sha256"] = canonical_json_hash(payload)

    directory = root / "results" / "indices"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "eval_indices.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"existing M4 eval indices conflict with the fixed contract: {path}"
            )
    else:
        json_dump(path, payload)
    return path, relative_indices, payload


def _resolve_matching_checkpoint(
    payload: dict[str, Any],
    result_path: Path,
) -> Path:
    expected_hash = payload.get("checkpoint_sha256")
    if not isinstance(expected_hash, str):
        raise ValueError("result has no checkpoint_sha256")

    stored = Path(str(payload.get("checkpoint", "")))
    candidates = (stored, result_path.parent / "checkpoint.npz")
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            checked.append(f"{candidate}: missing")
            continue
        actual_hash = sha256_file(candidate)
        if actual_hash == expected_hash:
            return candidate.resolve()
        checked.append(f"{candidate}: sha256={actual_hash}")
    raise ValueError(
        "no checkpoint matches the recorded hash; " + "; ".join(checked)
    )


def _validated_checkpoint_record(
    result_path: Path,
    *,
    spec: ModelSpec,
    seed: int,
    dataset_hash: str,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected_fields = {
        "experiment_id": spec.source_experiment_id,
        "config_id": spec.source_config_id,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "loss_id": "paper_squared",
        "verification": "artifacts-verified",
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_fields.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"result contract mismatch: {mismatches}")

    checkpoint_path = _resolve_matching_checkpoint(payload, result_path)
    theta, alpha, weights = load_checkpoint(checkpoint_path)
    expected_theta_shape = (1, spec.layers, 3)
    expected_alpha_shape = (1, spec.layers, 2)
    if theta.shape != expected_theta_shape:
        raise ValueError(
            f"theta shape {theta.shape} != expected {expected_theta_shape}"
        )
    if alpha.shape != expected_alpha_shape:
        raise ValueError(
            f"alpha shape {alpha.shape} != expected {expected_alpha_shape}"
        )
    if weights is not None:
        raise ValueError("M4 ordinary checkpoints must not contain weights")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(alpha)):
        raise ValueError("checkpoint contains non-finite parameters")

    return {
        "model_id": spec.model_id,
        "model_code": spec.model_code,
        "training_seed": seed,
        "source_experiment_id": spec.source_experiment_id,
        "source_config_id": spec.source_config_id,
        "source_result": str(result_path.resolve()),
        "source_result_sha256": sha256_file(result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_recorded_path": payload["checkpoint"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "theta_shape": list(theta.shape),
        "alpha_shape": list(alpha.shape),
        "dataset_hash": payload["dataset_hash"],
        "source_code_revision": payload.get("code_revision"),
    }


def discover_checkpoint_records(
    dataset_hash: str,
    *,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Find the latest hash-valid result for every fixed model/seed pair."""

    if raw_root is None:
        raw_root = ROOT / "results" / "raw"
    records: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            candidate_root = (
                raw_root
                / spec.source_experiment_id
                / spec.source_config_id
                / f"seed-{seed}"
            )
            candidates = sorted(candidate_root.glob("*/result.json"))
            valid: list[tuple[Path, dict[str, Any]]] = []
            failures: list[str] = []
            for candidate in candidates:
                try:
                    record = _validated_checkpoint_record(
                        candidate,
                        spec=spec,
                        seed=seed,
                        dataset_hash=dataset_hash,
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    failures.append(f"{candidate}: {exc}")
                    continue
                valid.append((candidate, record))

            if not valid:
                details = "\n".join(failures) if failures else "no result.json found"
                raise ValueError(
                    "missing verified frozen checkpoint for "
                    f"{spec.model_id}, seed={seed} under {candidate_root}\n{details}"
                )

            selected_path, selected_record = max(
                valid,
                key=lambda item: (item[0].stat().st_mtime_ns, str(item[0])),
            )
            selected_record["verified_candidate_count"] = len(valid)
            selected_record["selection_rule"] = "latest-valid-result-mtime"
            selected_record["selected_result"] = str(selected_path.resolve())
            records.append(selected_record)

    expected_count = len(MODEL_SPECS) * len(CONTROLLED_SEEDS)
    if len(records) != expected_count:
        raise AssertionError(
            f"checkpoint discovery returned {len(records)} != {expected_count}"
        )
    return records


def compute_exact_probabilities(
    checkpoint_records: list[dict[str, Any]],
    dataset: CircleDataset,
    eval_indices: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Evaluate frozen one-qubit ``P(0)`` values without any optimization."""

    points = dataset.test_x[eval_indices]
    labels = dataset.test_y[eval_indices]
    probability_rows = np.empty(
        (len(checkpoint_records), len(eval_indices)),
        dtype=np.float64,
    )
    exact_records: list[dict[str, Any]] = []

    for record_index, record in enumerate(checkpoint_records):
        checkpoint_path = Path(record["checkpoint"])
        if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed after discovery: {checkpoint_path}")
        theta, alpha, weights = load_checkpoint(checkpoint_path)
        if weights is not None:
            raise ValueError(f"weighted checkpoint is outside M4: {checkpoint_path}")

        max_normalization_error = 0.0
        for point_index, point in enumerate(points):
            probabilities = label_probabilities(theta, alpha, point, "n")
            if probabilities.shape != (2,) or not np.all(
                np.isfinite(probabilities)
            ):
                raise ValueError(
                    f"invalid exact probabilities from {checkpoint_path}"
                )
            normalization_error = abs(float(np.sum(probabilities)) - 1.0)
            max_normalization_error = max(
                max_normalization_error, normalization_error
            )
            probability_rows[record_index, point_index] = probabilities[0]

        if max_normalization_error > 1e-10:
            raise ValueError(
                "one-qubit probabilities are not normalized within 1e-10: "
                f"{checkpoint_path}, error={max_normalization_error}"
            )
        p0 = probability_rows[record_index]
        if np.any((p0 < -1e-12) | (p0 > 1.0 + 1e-12)):
            raise ValueError(f"exact P(0) lies outside [0, 1]: {checkpoint_path}")
        np.clip(p0, 0.0, 1.0, out=p0)
        exact_accuracy = float(np.mean(exact_predictions(p0) == labels))
        exact_records.append(
            {
                "model_id": record["model_id"],
                "model_code": record["model_code"],
                "training_seed": record["training_seed"],
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "point_count": int(len(eval_indices)),
                "exact_accuracy": exact_accuracy,
                "max_probability_normalization_error": max_normalization_error,
            }
        )
        print(
            "M4_EXACT_PROGRESS "
            + json.dumps(
                {
                    "model_id": record["model_id"],
                    "training_seed": record["training_seed"],
                    "exact_accuracy": exact_accuracy,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return probability_rows, exact_records


def campaign_rng(
    *,
    model_code: int,
    training_seed: int,
    repeat_index: int,
) -> tuple[np.random.Generator, dict[str, Any]]:
    """Derive one order-independent PCG64 campaign RNG."""

    entropy = (
        CAMPAIGN_MASTER_SEED,
        RNG_NAMESPACE,
        int(model_code),
        int(training_seed),
        int(repeat_index),
    )
    seed_sequence = np.random.SeedSequence(entropy)
    state_words = seed_sequence.generate_state(4, dtype=np.uint32)
    metadata = {
        "rng_bit_generator": "PCG64",
        "rng_entropy": ":".join(str(value) for value in entropy),
        "rng_state_words": ":".join(str(int(value)) for value in state_words),
    }
    return np.random.default_rng(seed_sequence), metadata


def _metric_row(
    summary: dict[str, Any],
    *,
    record: dict[str, Any],
    exact_accuracy: float,
    repeat_index: int,
    rng_metadata: dict[str, Any],
    method_id: str,
    policy: str,
    nominal_shots: int | None,
) -> dict[str, Any]:
    row = {
        "campaign_id": (
            f"{record['model_id']}-seed-{record['training_seed']}"
            f"-repeat-{repeat_index:03d}"
        ),
        "model_id": record["model_id"],
        "model_code": record["model_code"],
        "training_seed": record["training_seed"],
        "repeat_index": repeat_index,
        "repeat_number": repeat_index + 1,
        **rng_metadata,
        "method_id": method_id,
        "policy": policy,
        "nominal_shots": nominal_shots,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "exact_accuracy": exact_accuracy,
        "n_points": summary["n_points"],
    }
    for field in CAMPAIGN_METRIC_FIELDS:
        row[field] = summary[field]
    row["alpha"] = summary.get("alpha")
    row["cp_tail_probability"] = summary.get("cp_tail_probability")
    return row


def run_one_campaign(
    record: dict[str, Any],
    exact_p0: np.ndarray,
    labels: np.ndarray,
    *,
    exact_accuracy: float,
    repeat_index: int,
) -> list[dict[str, Any]]:
    """Return four scalar metric rows from one shared-prefix campaign."""

    rng, rng_metadata = campaign_rng(
        model_code=int(record["model_code"]),
        training_seed=int(record["training_seed"]),
        repeat_index=repeat_index,
    )
    campaign = run_shot_campaign(
        exact_p0,
        labels,
        rng,
        fixed_shots=DEFAULT_FIXED_SHOTS,
        adaptive_checkpoints=DEFAULT_ADAPTIVE_CHECKPOINTS,
        alpha=DEFAULT_ALPHA,
        include_pointwise=False,
    )
    rows = [
        _metric_row(
            campaign["fixed"][shots],
            record=record,
            exact_accuracy=exact_accuracy,
            repeat_index=repeat_index,
            rng_metadata=rng_metadata,
            method_id=f"fixed-{shots}",
            policy="fixed",
            nominal_shots=shots,
        )
        for shots in DEFAULT_FIXED_SHOTS
    ]
    rows.append(
        _metric_row(
            campaign["adaptive"],
            record=record,
            exact_accuracy=exact_accuracy,
            repeat_index=repeat_index,
            rng_metadata=rng_metadata,
            method_id="adaptive",
            policy="clopper-pearson-sequential",
            nominal_shots=None,
        )
    )
    return rows


def _mean_sample_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def aggregate_campaign_metrics(
    rows: list[dict[str, Any]],
    *,
    expected_repeats: int = CAMPAIGN_REPEATS,
) -> dict[str, Any]:
    """Aggregate repeats within seed, then aggregate five seed means."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model_id"]),
            int(row["training_seed"]),
            str(row["method_id"]),
        )
        grouped.setdefault(key, []).append(row)

    per_training_seed: list[dict[str, Any]] = []
    for (model_id, training_seed, method_id), group in sorted(grouped.items()):
        repeat_indices = {int(row["repeat_index"]) for row in group}
        if len(group) != expected_repeats or repeat_indices != set(
            range(expected_repeats)
        ):
            raise ValueError(
                "campaign repeat coverage mismatch for "
                f"{model_id}, seed={training_seed}, method={method_id}"
            )
        metric_stats = {
            field: _mean_sample_sd([float(row[field]) for row in group])
            for field in CAMPAIGN_METRIC_FIELDS
        }
        exact_accuracies = {float(row["exact_accuracy"]) for row in group}
        if len(exact_accuracies) != 1:
            raise ValueError("exact accuracy changed within a checkpoint")
        per_training_seed.append(
            {
                "model_id": model_id,
                "training_seed": training_seed,
                "method_id": method_id,
                "repeat_count": len(group),
                "exact_accuracy": exact_accuracies.pop(),
                "campaign_metrics": metric_stats,
            }
        )

    per_model: list[dict[str, Any]] = []
    model_method_groups: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    for row in per_training_seed:
        model_method_groups.setdefault(
            (row["model_id"], row["method_id"]), []
        ).append(row)
    for (model_id, method_id), group in sorted(model_method_groups.items()):
        seeds = {int(row["training_seed"]) for row in group}
        if seeds != set(CONTROLLED_SEEDS):
            raise ValueError(
                f"training seed coverage mismatch for {model_id}, {method_id}: "
                f"{sorted(seeds)}"
            )
        per_model.append(
            {
                "model_id": model_id,
                "method_id": method_id,
                "training_seed_count": len(group),
                "training_seed_metrics": {
                    field: _mean_sample_sd(
                        [
                            float(row["campaign_metrics"][field]["mean"])
                            for row in group
                        ]
                    )
                    for field in CAMPAIGN_METRIC_FIELDS
                },
            }
        )

    seed_lookup = {
        (row["model_id"], row["training_seed"], row["method_id"]): row
        for row in per_training_seed
    }
    model_lookup = {
        (row["model_id"], row["method_id"]): row for row in per_model
    }
    campaign_lookup = {
        (
            str(row["model_id"]),
            int(row["training_seed"]),
            int(row["repeat_index"]),
            str(row["method_id"]),
        ): row
        for row in rows
    }
    paired_accuracy_deltas: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for shots in DEFAULT_FIXED_SHOTS:
            fixed_method = f"fixed-{shots}"
            per_seed_deltas: list[dict[str, Any]] = []
            for seed in CONTROLLED_SEEDS:
                repeat_deltas = [
                    float(
                        campaign_lookup[
                            (spec.model_id, seed, repeat_index, "adaptive")
                        ]["accuracy"]
                    )
                    - float(
                        campaign_lookup[
                            (spec.model_id, seed, repeat_index, fixed_method)
                        ]["accuracy"]
                    )
                    for repeat_index in range(expected_repeats)
                ]
                per_seed_deltas.append(
                    {
                        "training_seed": seed,
                        "repeat_count": expected_repeats,
                        "adaptive_minus_fixed_accuracy": _mean_sample_sd(
                            repeat_deltas
                        ),
                    }
                )
            paired_accuracy_deltas.append(
                {
                    "model_id": spec.model_id,
                    "comparison": f"adaptive-minus-{fixed_method}",
                    "aggregation": (
                        "paired within campaign; mean over repeats within seed; "
                        "then mean and sample SD over training-seed means"
                    ),
                    "training_seed_delta": per_seed_deltas,
                    "across_training_seeds": _mean_sample_sd(
                        [
                            row["adaptive_minus_fixed_accuracy"]["mean"]
                            for row in per_seed_deltas
                        ]
                    ),
                }
            )

    adaptive_assessment: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        seed_checks: list[dict[str, Any]] = []
        for seed in CONTROLLED_SEEDS:
            adaptive = seed_lookup[(spec.model_id, seed, "adaptive")]
            fixed_2048 = seed_lookup[(spec.model_id, seed, "fixed-2048")]
            adaptive_accuracy = adaptive["campaign_metrics"]["accuracy"]["mean"]
            fixed_accuracy = fixed_2048["campaign_metrics"]["accuracy"]["mean"]
            adaptive_mean_shots = adaptive["campaign_metrics"]["mean_shots"]["mean"]
            seed_checks.append(
                {
                    "training_seed": seed,
                    "adaptive_accuracy_minus_fixed_2048": (
                        adaptive_accuracy - fixed_accuracy
                    ),
                    "adaptive_mean_shots": adaptive_mean_shots,
                    "shots_below_2048": adaptive_mean_shots < 2048.0,
                    "accuracy_drop_within_0_005": (
                        adaptive_accuracy - fixed_accuracy >= -0.005
                    ),
                    "passes_both_thresholds": (
                        adaptive_mean_shots < 2048.0
                        and adaptive_accuracy - fixed_accuracy >= -0.005
                    ),
                }
            )

        adaptive_model = model_lookup[(spec.model_id, "adaptive")]
        adaptive_accuracy = adaptive_model["training_seed_metrics"]["accuracy"][
            "mean"
        ]
        adaptive_total_shots = adaptive_model["training_seed_metrics"][
            "total_shots"
        ]["mean"]
        fixed_dominators: list[str] = []
        for shots in DEFAULT_FIXED_SHOTS:
            fixed = model_lookup[(spec.model_id, f"fixed-{shots}")]
            fixed_accuracy = fixed["training_seed_metrics"]["accuracy"]["mean"]
            fixed_total_shots = fixed["training_seed_metrics"]["total_shots"][
                "mean"
            ]
            dominates = (
                fixed_accuracy >= adaptive_accuracy
                and fixed_total_shots <= adaptive_total_shots
                and (
                    fixed_accuracy > adaptive_accuracy
                    or fixed_total_shots < adaptive_total_shots
                )
            )
            if dominates:
                fixed_dominators.append(f"fixed-{shots}")

        fixed_2048_model = model_lookup[(spec.model_id, "fixed-2048")]
        model_accuracy_delta = (
            adaptive_accuracy
            - fixed_2048_model["training_seed_metrics"]["accuracy"]["mean"]
        )
        adaptive_mean_shots = adaptive_model["training_seed_metrics"][
            "mean_shots"
        ]["mean"]
        seed_pass_count = sum(
            bool(row["passes_both_thresholds"]) for row in seed_checks
        )
        adaptive_assessment.append(
            {
                "model_id": spec.model_id,
                "fixed_methods_dominating_adaptive": fixed_dominators,
                "not_dominated_by_any_fixed": not fixed_dominators,
                "adaptive_mean_shots": adaptive_mean_shots,
                "mean_shots_below_2048": adaptive_mean_shots < 2048.0,
                "adaptive_accuracy_minus_fixed_2048": model_accuracy_delta,
                "accuracy_drop_within_0_005": model_accuracy_delta >= -0.005,
                "seed_threshold_pass_count": seed_pass_count,
                "at_least_four_seeds_pass": seed_pass_count >= 4,
                "adaptive_criteria_pass": (
                    not fixed_dominators
                    and adaptive_mean_shots < 2048.0
                    and model_accuracy_delta >= -0.005
                    and seed_pass_count >= 4
                ),
                "per_training_seed": seed_checks,
            }
        )

    return {
        "schema_version": 1,
        "aggregation_order": (
            "mean over 100 campaigns within each training seed, "
            "then mean and sample SD over five training seeds"
        ),
        "campaign_metric_fields": list(CAMPAIGN_METRIC_FIELDS),
        "per_training_seed": per_training_seed,
        "per_model": per_model,
        "paired_accuracy_deltas": paired_accuracy_deltas,
        "adaptive_thresholds": {
            "max_accuracy_drop_vs_fixed_2048": 0.005,
            "mean_shots_strictly_below": 2048,
            "minimum_passing_training_seeds": 4,
            "must_not_be_dominated_by_any_fixed": True,
        },
        "adaptive_assessment": adaptive_assessment,
    }


def _write_campaign_rows(
    handle: TextIO,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def run_m4_shot_evaluation() -> Path:
    """Execute the complete M4 evaluation without any optimizer runs."""

    dataset = make_circle_dataset()
    dataset_path = save_dataset(dataset)
    indices_path, eval_indices, indices_payload = eval_indices_artifact(dataset)
    checkpoint_records = discover_checkpoint_records(dataset.dataset_hash)
    source_hashes_before = evaluation_source_hashes()
    revision = git_revision()
    command = exact_command()

    run_dir = create_run_directory(EXPERIMENT_ID, CONFIG_ID, EVAL_SEED)
    checkpoint_records_path = run_dir / "checkpoint_records.json"
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    exact_path = run_dir / "exact_probabilities.npz"
    campaign_jsonl_path = run_dir / "campaign_metrics.jsonl"
    campaign_json_path = run_dir / "campaign_metrics.json"
    campaign_csv_path = run_dir / "campaign_metrics.csv"
    metrics_path = run_dir / "metrics.json"
    result_path = run_dir / "result.json"

    json_dump(checkpoint_records_path, checkpoint_records)
    json_dump(environment_path, environment_record())
    _write_text_exclusive(run_dir / "command.txt", command + "\n")
    config_payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_id": CONFIG_ID,
        "evidence_label": EVIDENCE_LABEL,
        "run_kind": "frozen-checkpoint-shot-evaluation",
        "optimizer_runs": 0,
        "dataset_hash": dataset.dataset_hash,
        "dataset_artifact": str(dataset_path),
        "eval_seed": EVAL_SEED,
        "sampling": "stratified-without-replacement",
        "points_per_class": POINTS_PER_CLASS,
        "eval_point_count": int(eval_indices.size),
        "fixed_shots": list(DEFAULT_FIXED_SHOTS),
        "adaptive_checkpoints": list(DEFAULT_ADAPTIVE_CHECKPOINTS),
        "adaptive_alpha": DEFAULT_ALPHA,
        "cp_tail_probability": DEFAULT_ALPHA
        / (2 * len(DEFAULT_ADAPTIVE_CHECKPOINTS)),
        "campaign_repeats_per_checkpoint": CAMPAIGN_REPEATS,
        "checkpoint_count": len(checkpoint_records),
        "rng_derivation": {
            "bit_generator": "PCG64",
            "seed_sequence_entropy": [
                "campaign_master_seed",
                "namespace",
                "model_code",
                "training_seed",
                "zero_based_repeat_index",
            ],
            "campaign_master_seed": CAMPAIGN_MASTER_SEED,
            "namespace": RNG_NAMESPACE,
            "model_code": {
                spec.model_id: spec.model_code for spec in MODEL_SPECS
            },
            "bernoulli_layout": (
                "rng.random((1000, 2048)) in C order; outcome0 iff draw < exact_p0"
            ),
            "prefix_contract": (
                "fixed 128/512/2048 and adaptive use the same campaign stream"
            ),
        },
        "eval_indices": str(indices_path),
        "eval_indices_file_sha256": sha256_file(indices_path),
        "eval_indices_content_sha256": indices_payload[
            "indices_content_sha256"
        ],
        "checkpoint_records": str(checkpoint_records_path),
        "checkpoint_records_sha256": sha256_file(checkpoint_records_path),
        "command": command,
        "code_revision": revision,
        "evaluation_source_sha256": source_hashes_before,
    }
    config_payload["config_fingerprint"] = canonical_json_hash(
        {
            key: value
            for key, value in config_payload.items()
            if key
            not in {
                "command",
                "code_revision",
                "dataset_artifact",
                "eval_indices",
                "checkpoint_records",
            }
        }
    )
    json_dump(config_path, config_payload)

    exact_p0, exact_records = compute_exact_probabilities(
        checkpoint_records,
        dataset,
        eval_indices,
    )
    labels = dataset.test_y[eval_indices].astype(np.int8)
    _save_npz_exclusive(
        exact_path,
        exact_p0=exact_p0,
        exact_prediction=np.vstack(
            [exact_predictions(row) for row in exact_p0]
        ),
        ground_truth_labels=labels,
        eval_test_relative_indices=np.asarray(eval_indices, dtype=np.int64),
        dataset_hash=np.asarray(dataset.dataset_hash),
        model_id=np.asarray(
            [record["model_id"] for record in checkpoint_records],
            dtype="U32",
        ),
        training_seed=np.asarray(
            [record["training_seed"] for record in checkpoint_records],
            dtype=np.int64,
        ),
        checkpoint_sha256=np.asarray(
            [record["checkpoint_sha256"] for record in checkpoint_records],
            dtype="U64",
        ),
    )

    all_campaign_rows: list[dict[str, Any]] = []
    with campaign_jsonl_path.open("x", encoding="utf-8") as campaign_handle:
        for record_index, (record, exact_record) in enumerate(
            zip(checkpoint_records, exact_records)
        ):
            for repeat_index in range(CAMPAIGN_REPEATS):
                rows = run_one_campaign(
                    record,
                    exact_p0[record_index],
                    labels,
                    exact_accuracy=exact_record["exact_accuracy"],
                    repeat_index=repeat_index,
                )
                _write_campaign_rows(campaign_handle, rows)
                all_campaign_rows.extend(rows)
                if (
                    repeat_index == 0
                    or (repeat_index + 1) % PROGRESS_EVERY == 0
                ):
                    print(
                        "M4_CAMPAIGN_PROGRESS "
                        + json.dumps(
                            {
                                "model_id": record["model_id"],
                                "training_seed": record["training_seed"],
                                "completed_repeats": repeat_index + 1,
                                "total_repeats": CAMPAIGN_REPEATS,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    expected_campaign_count = len(checkpoint_records) * CAMPAIGN_REPEATS
    expected_row_count = expected_campaign_count * (
        len(DEFAULT_FIXED_SHOTS) + 1
    )
    if len(all_campaign_rows) != expected_row_count:
        raise AssertionError(
            f"campaign row count {len(all_campaign_rows)} != {expected_row_count}"
        )
    json_dump(campaign_json_path, all_campaign_rows)
    _write_csv_exclusive(campaign_csv_path, all_campaign_rows)

    metrics_payload = aggregate_campaign_metrics(all_campaign_rows)
    metrics_payload.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_id": CONFIG_ID,
            "dataset_hash": dataset.dataset_hash,
            "eval_indices_file_sha256": sha256_file(indices_path),
            "exact_checkpoint_metrics": exact_records,
            "campaign_count": expected_campaign_count,
            "campaign_metric_row_count": expected_row_count,
        }
    )
    json_dump(metrics_path, metrics_payload)

    source_hashes_after = evaluation_source_hashes()
    source_stable = source_hashes_after == source_hashes_before
    verification = (
        "artifacts-verified" if source_stable else "code-changed-during-run"
    )
    result_payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_id": CONFIG_ID,
        "evidence_label": EVIDENCE_LABEL,
        "verification": verification,
        "optimizer_runs": 0,
        "nfev": 0,
        "dataset_hash": dataset.dataset_hash,
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "checkpoint_count": len(checkpoint_records),
        "campaign_repeats_per_checkpoint": CAMPAIGN_REPEATS,
        "campaign_count": expected_campaign_count,
        "campaign_metric_row_count": expected_row_count,
        "eval_indices": str(indices_path),
        "eval_indices_file_sha256": sha256_file(indices_path),
        "eval_indices_content_sha256": indices_payload[
            "indices_content_sha256"
        ],
        "checkpoint_records": str(checkpoint_records_path),
        "checkpoint_records_sha256": sha256_file(checkpoint_records_path),
        "exact_probabilities": str(exact_path),
        "exact_probabilities_sha256": sha256_file(exact_path),
        "campaign_metrics_jsonl": str(campaign_jsonl_path),
        "campaign_metrics_jsonl_sha256": sha256_file(campaign_jsonl_path),
        "campaign_metrics_json": str(campaign_json_path),
        "campaign_metrics_json_sha256": sha256_file(campaign_json_path),
        "campaign_metrics_csv": str(campaign_csv_path),
        "campaign_metrics_csv_sha256": sha256_file(campaign_csv_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "environment": str(environment_path),
        "environment_sha256": sha256_file(environment_path),
        "command": command,
        "code_revision": revision,
        "evaluation_source_sha256_before": source_hashes_before,
        "evaluation_source_sha256_after": source_hashes_after,
        "evaluation_source_stable": source_stable,
        "raw_result_path": str(run_dir),
    }
    json_dump(result_path, result_payload)
    print("RESULT " + json.dumps(result_payload, sort_keys=True), flush=True)
    if not source_stable:
        raise SystemExit(
            "M4 artifacts were preserved, but evaluation source changed during run"
        )
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run M4 fixed/adaptive shots on ten verified frozen checkpoints"
        )
    )
    parser.parse_args()
    run_m4_shot_evaluation()


if __name__ == "__main__":
    main()
