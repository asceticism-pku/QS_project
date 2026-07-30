#!/usr/bin/env python3
"""Validate one completed M4 raw run and create a non-overwriting summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.run_shot_evaluation import (  # noqa: E402
    CAMPAIGN_METRIC_FIELDS,
    CAMPAIGN_REPEATS,
    CONFIG_ID,
    EVAL_SEED,
    EXPERIMENT_ID,
    MODEL_SPECS,
    aggregate_campaign_metrics,
)
from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    TRAIN_COUNT,
    canonical_json_hash,
    git_revision,
    json_dump,
    sha256_file,
)
from qs_project.shots import (  # noqa: E402
    DEFAULT_ADAPTIVE_CHECKPOINTS,
    DEFAULT_ALPHA,
    DEFAULT_FIXED_SHOTS,
    exact_predictions,
)


EXPECTED_CHECKPOINTS = 10
EXPECTED_CAMPAIGNS = 1000
EXPECTED_CAMPAIGN_ROWS = 4000
EXPECTED_EVAL_POINTS = 1000
METHOD_ORDER = tuple(
    [f"fixed-{shots}" for shots in DEFAULT_FIXED_SHOTS] + ["adaptive"]
)
MODEL_ORDER = tuple(spec.model_id for spec in MODEL_SPECS)
MODEL_SPEC_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}


@dataclass(frozen=True)
class ValidatedM4:
    result_path: Path
    result: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, Any]
    campaign_rows: list[dict[str, Any]]
    checkpoint_records: list[dict[str, Any]]
    exact_metadata: list[dict[str, Any]]
    eval_indices: dict[str, Any]
    input_artifacts: dict[str, dict[str, str]]


def exact_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_keys(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def referenced_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checked_artifact(
    payload: dict[str, Any],
    path_key: str,
    hash_key: str,
    *,
    label: str,
) -> Path:
    require_keys(payload, (path_key, hash_key), label=label)
    path = referenced_path(str(payload[path_key]))
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    expected = str(payload[hash_key])
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch for {path}: {actual} != {expected}"
        )
    return path


def latest_raw_result() -> Path:
    root = (
        ROOT
        / "results"
        / "raw"
        / EXPERIMENT_ID
        / CONFIG_ID
        / f"seed-{EVAL_SEED}"
    )
    candidates: list[Path] = []
    for path in sorted(root.glob("*/result.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("experiment_id") == EXPERIMENT_ID
            and payload.get("config_id") == CONFIG_ID
            and payload.get("verification") == "artifacts-verified"
        ):
            candidates.append(path)
    if not candidates:
        raise ValueError(f"no completed verified M4 result under {root}")
    return max(candidates, key=lambda path: path.parent.name)


def _validate_result_contract(
    result_path: Path,
    result: dict[str, Any],
) -> None:
    require_keys(
        result,
        (
            "experiment_id",
            "config_id",
            "evidence_label",
            "verification",
            "optimizer_runs",
            "nfev",
            "dataset_hash",
            "checkpoint_count",
            "campaign_repeats_per_checkpoint",
            "campaign_count",
            "campaign_metric_row_count",
            "raw_result_path",
        ),
        label="M4 result",
    )
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "config_id": CONFIG_ID,
        "evidence_label": "shot-simulation",
        "verification": "artifacts-verified",
        "optimizer_runs": 0,
        "nfev": 0,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "campaign_repeats_per_checkpoint": CAMPAIGN_REPEATS,
        "campaign_count": EXPECTED_CAMPAIGNS,
        "campaign_metric_row_count": EXPECTED_CAMPAIGN_ROWS,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"M4 result contract mismatch: {mismatches}")
    if referenced_path(str(result["raw_result_path"])).resolve() != (
        result_path.parent.resolve()
    ):
        raise ValueError("M4 raw_result_path does not match result.json parent")


def _validate_config(
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "config_id": CONFIG_ID,
        "evidence_label": "shot-simulation",
        "run_kind": "frozen-checkpoint-shot-evaluation",
        "optimizer_runs": 0,
        "dataset_hash": result["dataset_hash"],
        "eval_seed": EVAL_SEED,
        "points_per_class": 500,
        "eval_point_count": EXPECTED_EVAL_POINTS,
        "fixed_shots": list(DEFAULT_FIXED_SHOTS),
        "adaptive_checkpoints": list(DEFAULT_ADAPTIVE_CHECKPOINTS),
        "adaptive_alpha": DEFAULT_ALPHA,
        "campaign_repeats_per_checkpoint": CAMPAIGN_REPEATS,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"M4 config contract mismatch: {mismatches}")
    if not math.isclose(
        float(config.get("cp_tail_probability", math.nan)),
        DEFAULT_ALPHA / 6,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("M4 config has the wrong CP tail probability")

    require_keys(config, ("config_fingerprint",), label="M4 config")
    fingerprint_payload = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "command",
            "code_revision",
            "dataset_artifact",
            "eval_indices",
            "checkpoint_records",
            "config_fingerprint",
        }
    }
    actual_fingerprint = canonical_json_hash(fingerprint_payload)
    if actual_fingerprint != config["config_fingerprint"]:
        raise ValueError("M4 config fingerprint mismatch")


def _validate_eval_indices(
    path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("eval_indices.json must contain an object")
    require_keys(
        payload,
        (
            "purpose",
            "eval_seed",
            "sampling",
            "per_class",
            "point_count",
            "index_space",
            "test_relative_indices",
            "dataset_global_indices",
            "labels",
            "dataset_hash",
            "indices_content_sha256",
        ),
        label="M4 eval indices",
    )
    content_payload = {
        key: value
        for key, value in payload.items()
        if key != "indices_content_sha256"
    }
    if canonical_json_hash(content_payload) != payload["indices_content_sha256"]:
        raise ValueError("M4 eval indices content hash mismatch")
    if payload["indices_content_sha256"] != result[
        "eval_indices_content_sha256"
    ]:
        raise ValueError("result/eval-indices content hash mismatch")
    if (
        payload["purpose"] != "M4-fixed-adaptive-shots"
        or payload["eval_seed"] != EVAL_SEED
        or payload["sampling"] != "stratified-without-replacement"
        or payload["per_class"] != 500
        or payload["point_count"] != EXPECTED_EVAL_POINTS
        or payload["index_space"] != "test-set-relative"
        or payload["dataset_hash"] != result["dataset_hash"]
    ):
        raise ValueError("M4 eval indices contract mismatch")

    relative = np.asarray(payload["test_relative_indices"], dtype=np.int64)
    global_indices = np.asarray(payload["dataset_global_indices"], dtype=np.int64)
    labels = np.asarray(payload["labels"], dtype=np.int8)
    if (
        relative.shape != (EXPECTED_EVAL_POINTS,)
        or global_indices.shape != (EXPECTED_EVAL_POINTS,)
        or labels.shape != (EXPECTED_EVAL_POINTS,)
    ):
        raise ValueError("M4 eval indices arrays have the wrong shape")
    if np.unique(relative).size != EXPECTED_EVAL_POINTS:
        raise ValueError("M4 eval indices contain duplicates")
    if not np.array_equal(global_indices, relative + TRAIN_COUNT):
        raise ValueError("M4 global and test-relative indices disagree")
    if np.count_nonzero(labels == 0) != 500 or np.count_nonzero(labels == 1) != 500:
        raise ValueError("M4 eval labels are not 500 points per class")
    return payload


def _validate_checkpoint_records(
    records: Any,
    *,
    dataset_hash: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != EXPECTED_CHECKPOINTS:
        raise ValueError("M4 checkpoint_records must contain exactly 10 records")
    expected_keys = {
        (spec.model_id, seed)
        for spec in MODEL_SPECS
        for seed in CONTROLLED_SEEDS
    }
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"M4 checkpoint record {index} is not an object")
        require_keys(
            record,
            (
                "model_id",
                "model_code",
                "training_seed",
                "source_experiment_id",
                "source_config_id",
                "source_result",
                "source_result_sha256",
                "checkpoint",
                "checkpoint_sha256",
                "theta_shape",
                "alpha_shape",
                "dataset_hash",
            ),
            label=f"M4 checkpoint record {index}",
        )
        model_id = str(record["model_id"])
        seed = int(record["training_seed"])
        key = (model_id, seed)
        if key in seen:
            raise ValueError(f"duplicate M4 checkpoint record: {key}")
        seen.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected M4 checkpoint record: {key}")
        spec = MODEL_SPEC_BY_ID[model_id]
        if (
            int(record["model_code"]) != spec.model_code
            or record["source_experiment_id"] != spec.source_experiment_id
            or record["source_config_id"] != spec.source_config_id
            or record["theta_shape"] != [1, spec.layers, 3]
            or record["alpha_shape"] != [1, spec.layers, 2]
            or record["dataset_hash"] != dataset_hash
        ):
            raise ValueError(f"M4 checkpoint identity mismatch: {key}")

        checkpoint = referenced_path(str(record["checkpoint"]))
        if not checkpoint.is_file():
            raise ValueError(f"M4 checkpoint is missing: {checkpoint}")
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"M4 checkpoint hash mismatch: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as checkpoint_data:
            if set(checkpoint_data.files) != {"theta", "alpha"}:
                raise ValueError(f"unexpected checkpoint arrays: {checkpoint}")
            if list(checkpoint_data["theta"].shape) != record["theta_shape"]:
                raise ValueError(f"theta shape mismatch: {checkpoint}")
            if list(checkpoint_data["alpha"].shape) != record["alpha_shape"]:
                raise ValueError(f"alpha shape mismatch: {checkpoint}")

        source_result = referenced_path(str(record["source_result"]))
        if not source_result.is_file():
            raise ValueError(f"source result is missing: {source_result}")
        if sha256_file(source_result) != record["source_result_sha256"]:
            raise ValueError(f"source result hash mismatch: {source_result}")
    if seen != expected_keys:
        raise ValueError(f"M4 checkpoint coverage mismatch: {sorted(seen)}")
    return records


def _validate_exact_probabilities(
    path: Path,
    *,
    records: list[dict[str, Any]],
    eval_indices: dict[str, Any],
    exact_metrics: Any,
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        expected_arrays = {
            "exact_p0",
            "exact_prediction",
            "ground_truth_labels",
            "eval_test_relative_indices",
            "dataset_hash",
            "model_id",
            "training_seed",
            "checkpoint_sha256",
        }
        if set(data.files) != expected_arrays:
            raise ValueError(
                f"exact probability arrays mismatch: {sorted(data.files)}"
            )
        p0 = np.asarray(data["exact_p0"], dtype=np.float64)
        prediction = np.asarray(data["exact_prediction"], dtype=np.int8)
        labels = np.asarray(data["ground_truth_labels"], dtype=np.int8)
        indices = np.asarray(data["eval_test_relative_indices"], dtype=np.int64)
        dataset_hash = str(np.asarray(data["dataset_hash"]).item())
        model_ids = np.asarray(data["model_id"]).astype(str).tolist()
        seeds = np.asarray(data["training_seed"], dtype=np.int64).tolist()
        checkpoint_hashes = (
            np.asarray(data["checkpoint_sha256"]).astype(str).tolist()
        )

    if p0.shape != (EXPECTED_CHECKPOINTS, EXPECTED_EVAL_POINTS):
        raise ValueError(f"exact_p0 has wrong shape: {p0.shape}")
    if prediction.shape != p0.shape:
        raise ValueError("exact prediction shape does not match exact_p0")
    if labels.shape != (EXPECTED_EVAL_POINTS,):
        raise ValueError("exact artifact labels have the wrong shape")
    if not np.all(np.isfinite(p0)) or np.any((p0 < 0.0) | (p0 > 1.0)):
        raise ValueError("exact_p0 contains invalid probabilities")
    expected_prediction = np.vstack([exact_predictions(row) for row in p0])
    if not np.array_equal(prediction, expected_prediction):
        raise ValueError("exact predictions do not match exact_p0")
    if not np.array_equal(
        labels, np.asarray(eval_indices["labels"], dtype=np.int8)
    ):
        raise ValueError("exact artifact labels do not match eval indices")
    if not np.array_equal(
        indices,
        np.asarray(eval_indices["test_relative_indices"], dtype=np.int64),
    ):
        raise ValueError("exact artifact indices do not match eval indices")
    if dataset_hash != eval_indices["dataset_hash"]:
        raise ValueError("exact artifact dataset hash mismatch")
    if model_ids != [str(record["model_id"]) for record in records]:
        raise ValueError("exact artifact model order mismatch")
    if seeds != [int(record["training_seed"]) for record in records]:
        raise ValueError("exact artifact seed order mismatch")
    if checkpoint_hashes != [
        str(record["checkpoint_sha256"]) for record in records
    ]:
        raise ValueError("exact artifact checkpoint order mismatch")

    if not isinstance(exact_metrics, list) or len(exact_metrics) != len(records):
        raise ValueError("metrics exact_checkpoint_metrics has wrong length")
    normalized: list[dict[str, Any]] = []
    for index, (record, metric) in enumerate(zip(records, exact_metrics)):
        if not isinstance(metric, dict):
            raise ValueError(f"exact checkpoint metric {index} is not an object")
        exact_accuracy = float(np.mean(prediction[index] == labels))
        if (
            metric.get("model_id") != record["model_id"]
            or int(metric.get("training_seed", -1)) != record["training_seed"]
            or metric.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or int(metric.get("point_count", -1)) != EXPECTED_EVAL_POINTS
            or not math.isclose(
                float(metric.get("exact_accuracy", math.nan)),
                exact_accuracy,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"exact checkpoint metric mismatch at index {index}")
        normalized.append(
            {
                "model_id": record["model_id"],
                "training_seed": record["training_seed"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "exact_accuracy": exact_accuracy,
            }
        )
    return normalized


def _load_campaign_representations(
    *,
    json_path: Path,
    jsonl_path: Path,
    csv_path: Path,
) -> list[dict[str, Any]]:
    json_rows = read_json(json_path)
    if not isinstance(json_rows, list):
        raise ValueError("campaign_metrics.json must contain a list")
    jsonl_rows: list[Any] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"blank line in campaign JSONL at {line_number}"
                )
            jsonl_rows.append(json.loads(line))
    if json_rows != jsonl_rows:
        raise ValueError("campaign JSON and JSONL rows differ")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        csv_fields = reader.fieldnames
    if len(json_rows) != len(csv_rows):
        raise ValueError("campaign CSV and JSON row counts differ")
    if json_rows:
        if csv_fields is None or set(csv_fields) != set(json_rows[0]):
            raise ValueError("campaign CSV and JSON fields differ")
        for row_index, (json_row, csv_row) in enumerate(
            zip(json_rows, csv_rows)
        ):
            if not isinstance(json_row, dict):
                raise ValueError(f"campaign JSON row {row_index} is not an object")
            expected_csv = {
                key: (
                    ""
                    if json_row[key] is None
                    else str(json_row[key])
                )
                for key in csv_fields
            }
            if csv_row != expected_csv:
                raise ValueError(
                    f"campaign CSV differs from JSON at row {row_index}"
                )
    return json_rows


def _validate_campaign_rows(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    exact_metadata: list[dict[str, Any]],
) -> None:
    if len(rows) != EXPECTED_CAMPAIGN_ROWS:
        raise ValueError(
            f"M4 campaign row count {len(rows)} != {EXPECTED_CAMPAIGN_ROWS}"
        )
    record_map = {
        (str(record["model_id"]), int(record["training_seed"])): record
        for record in records
    }
    exact_map = {
        (str(record["model_id"]), int(record["training_seed"])): record
        for record in exact_metadata
    }
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    tuple_methods: set[tuple[str, int, int, str]] = set()

    for row_index, row in enumerate(rows):
        require_keys(
            row,
            (
                "campaign_id",
                "model_id",
                "model_code",
                "training_seed",
                "repeat_index",
                "rng_bit_generator",
                "rng_entropy",
                "rng_state_words",
                "method_id",
                "policy",
                "checkpoint",
                "checkpoint_sha256",
                "exact_accuracy",
                "n_points",
                *CAMPAIGN_METRIC_FIELDS,
            ),
            label=f"campaign row {row_index}",
        )
        model_id = str(row["model_id"])
        seed = int(row["training_seed"])
        repeat_index = int(row["repeat_index"])
        method_id = str(row["method_id"])
        key = (model_id, seed)
        if key not in record_map:
            raise ValueError(f"campaign row has unexpected checkpoint: {key}")
        if not 0 <= repeat_index < CAMPAIGN_REPEATS:
            raise ValueError(f"campaign repeat index out of range: {repeat_index}")
        tuple_key = (model_id, seed, repeat_index, method_id)
        if tuple_key in tuple_methods:
            raise ValueError(f"duplicate campaign method row: {tuple_key}")
        tuple_methods.add(tuple_key)
        if method_id not in METHOD_ORDER:
            raise ValueError(f"unexpected M4 method: {method_id}")

        record = record_map[key]
        if (
            int(row["model_code"]) != int(record["model_code"])
            or referenced_path(str(row["checkpoint"])).resolve()
            != referenced_path(str(record["checkpoint"])).resolve()
            or row["checkpoint_sha256"] != record["checkpoint_sha256"]
            or int(row["n_points"]) != EXPECTED_EVAL_POINTS
            or row["rng_bit_generator"] != "PCG64"
            or row["rng_entropy"]
            != f"2026:4:{record['model_code']}:{seed}:{repeat_index}"
            or not math.isclose(
                float(row["exact_accuracy"]),
                float(exact_map[key]["exact_accuracy"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"campaign checkpoint/RNG contract mismatch: {tuple_key}")

        accuracy = float(row["accuracy"])
        flip_rate = float(row["exact_label_flip_rate"])
        if not 0.0 <= accuracy <= 1.0 or not 0.0 <= flip_rate <= 1.0:
            raise ValueError(f"campaign rate outside [0, 1]: {tuple_key}")
        if method_id.startswith("fixed-"):
            shots = int(method_id.removeprefix("fixed-"))
            if (
                row["policy"] != "fixed"
                or int(row["nominal_shots"]) != shots
                or int(row["total_shots"]) != EXPECTED_EVAL_POINTS * shots
                or int(row["cap_hit"]) != 0
                or float(row["cap_hit_fraction"]) != 0.0
                or any(
                    float(row[field]) != float(shots)
                    for field in (
                        "mean_shots",
                        "median_shots",
                        "p90_shots",
                        "p95_shots",
                    )
                )
                or row["alpha"] is not None
                or row["cp_tail_probability"] is not None
            ):
                raise ValueError(f"fixed-shot metric contract mismatch: {tuple_key}")
        else:
            stop_fractions = [
                float(row[f"stop_fraction_{shots}"])
                for shots in DEFAULT_ADAPTIVE_CHECKPOINTS
            ]
            expected_mean_shots = sum(
                shots * fraction
                for shots, fraction in zip(
                    DEFAULT_ADAPTIVE_CHECKPOINTS, stop_fractions
                )
            )
            if (
                row["policy"] != "clopper-pearson-sequential"
                or row["nominal_shots"] is not None
                or not math.isclose(
                    sum(stop_fractions), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    float(row["mean_shots"]),
                    expected_mean_shots,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(row["total_shots"]),
                    EXPECTED_EVAL_POINTS * float(row["mean_shots"]),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    float(row["cap_hit_fraction"]),
                    int(row["cap_hit"]) / EXPECTED_EVAL_POINTS,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(row["alpha"]),
                    DEFAULT_ALPHA,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    float(row["cp_tail_probability"]),
                    DEFAULT_ALPHA / 6,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"adaptive-shot metric contract mismatch: {tuple_key}")
        grouped.setdefault((model_id, seed, repeat_index), []).append(row)

    expected_campaign_keys = {
        (spec.model_id, seed, repeat_index)
        for spec in MODEL_SPECS
        for seed in CONTROLLED_SEEDS
        for repeat_index in range(CAMPAIGN_REPEATS)
    }
    if set(grouped) != expected_campaign_keys:
        raise ValueError("M4 campaign coverage is not exactly 1000 campaigns")
    for campaign_key, campaign_rows in grouped.items():
        if {str(row["method_id"]) for row in campaign_rows} != set(METHOD_ORDER):
            raise ValueError(f"campaign method coverage mismatch: {campaign_key}")
        rng_records = {
            (
                row["rng_bit_generator"],
                row["rng_entropy"],
                row["rng_state_words"],
            )
            for row in campaign_rows
        }
        if len(rng_records) != 1:
            raise ValueError(f"campaign methods do not share RNG: {campaign_key}")


def _validate_metrics(
    metrics: Any,
    *,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    exact_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain an object")
    if (
        metrics.get("experiment_id") != EXPERIMENT_ID
        or metrics.get("config_id") != CONFIG_ID
        or metrics.get("dataset_hash") != result["dataset_hash"]
        or metrics.get("eval_indices_file_sha256")
        != result["eval_indices_file_sha256"]
        or metrics.get("campaign_count") != EXPECTED_CAMPAIGNS
        or metrics.get("campaign_metric_row_count") != EXPECTED_CAMPAIGN_ROWS
    ):
        raise ValueError("metrics.json metadata contract mismatch")

    recomputed = aggregate_campaign_metrics(rows)
    for key, expected in recomputed.items():
        if metrics.get(key) != expected:
            raise ValueError(f"metrics.json aggregate mismatch: {key}")
    if len(metrics.get("exact_checkpoint_metrics", [])) != len(exact_metadata):
        raise ValueError("metrics.json exact checkpoint count mismatch")
    return metrics


def validate_m4_raw(result_path: Path) -> ValidatedM4:
    """Validate hashes, cross-format equality, coverage, and aggregation."""

    result_path = Path(result_path)
    if result_path.is_dir():
        result_path = result_path / "result.json"
    if not result_path.is_file():
        raise ValueError(f"M4 result does not exist: {result_path}")
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("M4 result.json must contain an object")
    _validate_result_contract(result_path, result)

    artifact_specs = {
        "config": ("config", "config_sha256"),
        "environment": ("environment", "environment_sha256"),
        "dataset": ("dataset_artifact", "dataset_artifact_sha256"),
        "eval_indices": ("eval_indices", "eval_indices_file_sha256"),
        "checkpoint_records": (
            "checkpoint_records",
            "checkpoint_records_sha256",
        ),
        "exact_probabilities": (
            "exact_probabilities",
            "exact_probabilities_sha256",
        ),
        "campaign_jsonl": (
            "campaign_metrics_jsonl",
            "campaign_metrics_jsonl_sha256",
        ),
        "campaign_json": (
            "campaign_metrics_json",
            "campaign_metrics_json_sha256",
        ),
        "campaign_csv": (
            "campaign_metrics_csv",
            "campaign_metrics_csv_sha256",
        ),
        "metrics": ("metrics", "metrics_sha256"),
    }
    paths: dict[str, Path] = {}
    input_artifacts = {
        "result": {
            "path": str(result_path.resolve()),
            "sha256": sha256_file(result_path),
        }
    }
    for label, (path_key, hash_key) in artifact_specs.items():
        path = checked_artifact(
            result,
            path_key,
            hash_key,
            label=f"M4 {label}",
        )
        paths[label] = path
        input_artifacts[label] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }

    config = read_json(paths["config"])
    if not isinstance(config, dict):
        raise ValueError("M4 config.json must contain an object")
    _validate_config(config, result)
    eval_indices = _validate_eval_indices(paths["eval_indices"], result)
    checkpoint_records = _validate_checkpoint_records(
        read_json(paths["checkpoint_records"]),
        dataset_hash=str(result["dataset_hash"]),
    )
    metrics = read_json(paths["metrics"])
    if not isinstance(metrics, dict):
        raise ValueError("M4 metrics.json must contain an object")
    exact_metadata = _validate_exact_probabilities(
        paths["exact_probabilities"],
        records=checkpoint_records,
        eval_indices=eval_indices,
        exact_metrics=metrics.get("exact_checkpoint_metrics"),
    )
    campaign_rows = _load_campaign_representations(
        json_path=paths["campaign_json"],
        jsonl_path=paths["campaign_jsonl"],
        csv_path=paths["campaign_csv"],
    )
    _validate_campaign_rows(
        campaign_rows,
        records=checkpoint_records,
        exact_metadata=exact_metadata,
    )
    metrics = _validate_metrics(
        metrics,
        rows=campaign_rows,
        result=result,
        exact_metadata=exact_metadata,
    )
    return ValidatedM4(
        result_path=result_path.resolve(),
        result=result,
        config=config,
        metrics=metrics,
        campaign_rows=campaign_rows,
        checkpoint_records=checkpoint_records,
        exact_metadata=exact_metadata,
        eval_indices=eval_indices,
        input_artifacts=input_artifacts,
    )


def _flatten_stats(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
) -> None:
    for field in CAMPAIGN_METRIC_FIELDS:
        stats = source[field]
        row[f"{field}_mean"] = stats["mean"]
        row[f"{field}_sample_sd"] = stats["sample_sd"]


def per_model_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        (str(item["model_id"]), str(item["method_id"])): item
        for item in metrics["per_model"]
    }
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for method_id in METHOD_ORDER:
            item = lookup[(model_id, method_id)]
            row: dict[str, Any] = {
                "model_id": model_id,
                "method_id": method_id,
                "training_seed_count": item["training_seed_count"],
            }
            _flatten_stats(row, source=item["training_seed_metrics"])
            rows.append(row)
    return rows


def per_training_seed_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(item["model_id"]),
            int(item["training_seed"]),
            str(item["method_id"]),
        ): item
        for item in metrics["per_training_seed"]
    }
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for seed in CONTROLLED_SEEDS:
            for method_id in METHOD_ORDER:
                item = lookup[(model_id, seed, method_id)]
                row: dict[str, Any] = {
                    "model_id": model_id,
                    "training_seed": seed,
                    "method_id": method_id,
                    "repeat_count": item["repeat_count"],
                    "exact_accuracy": item["exact_accuracy"],
                }
                _flatten_stats(row, source=item["campaign_metrics"])
                rows.append(row)
    return rows


def paired_accuracy_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        (str(item["model_id"]), str(item["comparison"])): item
        for item in metrics["paired_accuracy_deltas"]
    }
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for shots in DEFAULT_FIXED_SHOTS:
            comparison = f"adaptive-minus-fixed-{shots}"
            item = lookup[(model_id, comparison)]
            seed_lookup = {
                int(seed_row["training_seed"]): seed_row
                for seed_row in item["training_seed_delta"]
            }
            for seed in CONTROLLED_SEEDS:
                seed_row = seed_lookup[seed]
                rows.append(
                    {
                        "model_id": model_id,
                        "comparison": comparison,
                        "training_seed": seed,
                        "repeat_count": seed_row["repeat_count"],
                        "paired_accuracy_delta_mean": seed_row[
                            "adaptive_minus_fixed_accuracy"
                        ]["mean"],
                        "paired_accuracy_delta_sample_sd": seed_row[
                            "adaptive_minus_fixed_accuracy"
                        ]["sample_sd"],
                        "across_training_seeds_mean": item[
                            "across_training_seeds"
                        ]["mean"],
                        "across_training_seeds_sample_sd": item[
                            "across_training_seeds"
                        ]["sample_sd"],
                    }
                )
    return rows


def adaptive_assessment_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        str(item["model_id"]): item for item in metrics["adaptive_assessment"]
    }
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        item = lookup[model_id]
        rows.append(
            {
                "scope": "model",
                "model_id": model_id,
                "training_seed": None,
                "fixed_methods_dominating_adaptive": ";".join(
                    item["fixed_methods_dominating_adaptive"]
                ),
                "not_dominated_by_any_fixed": item[
                    "not_dominated_by_any_fixed"
                ],
                "adaptive_mean_shots": item["adaptive_mean_shots"],
                "mean_shots_below_2048": item["mean_shots_below_2048"],
                "adaptive_accuracy_minus_fixed_2048": item[
                    "adaptive_accuracy_minus_fixed_2048"
                ],
                "accuracy_drop_within_0_005": item[
                    "accuracy_drop_within_0_005"
                ],
                "seed_threshold_pass_count": item["seed_threshold_pass_count"],
                "at_least_four_seeds_pass": item["at_least_four_seeds_pass"],
                "passes_both_thresholds": None,
                "adaptive_criteria_pass": item["adaptive_criteria_pass"],
            }
        )
        for seed_item in item["per_training_seed"]:
            rows.append(
                {
                    "scope": "training-seed",
                    "model_id": model_id,
                    "training_seed": seed_item["training_seed"],
                    "fixed_methods_dominating_adaptive": None,
                    "not_dominated_by_any_fixed": None,
                    "adaptive_mean_shots": seed_item["adaptive_mean_shots"],
                    "mean_shots_below_2048": seed_item["shots_below_2048"],
                    "adaptive_accuracy_minus_fixed_2048": seed_item[
                        "adaptive_accuracy_minus_fixed_2048"
                    ],
                    "accuracy_drop_within_0_005": seed_item[
                        "accuracy_drop_within_0_005"
                    ],
                    "seed_threshold_pass_count": None,
                    "at_least_four_seeds_pass": None,
                    "passes_both_thresholds": seed_item[
                        "passes_both_thresholds"
                    ],
                    "adaptive_criteria_pass": None,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sha256sums(output: Path, filenames: tuple[str, ...]) -> Path:
    path = output / "SHA256SUMS"
    with path.open("x", encoding="utf-8") as handle:
        for filename in filenames:
            artifact = output / filename
            if not artifact.is_file():
                raise ValueError(
                    f"cannot checksum missing summary artifact: {artifact}"
                )
            handle.write(f"{sha256_file(artifact)}  {filename}\n")
    return path


def unique_summary_dir(*, output_root: Path | None = None) -> Path:
    if output_root is None:
        output_root = ROOT / "results" / "summary" / EXPERIMENT_ID
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _recheck_input_hashes(validated: ValidatedM4) -> None:
    for label, record in validated.input_artifacts.items():
        path = Path(record["path"])
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"M4 input changed during summary: {label}, {path}"
            )


def write_m4_summary(
    validated: ValidatedM4,
    *,
    output_root: Path | None = None,
) -> Path:
    output = unique_summary_dir(output_root=output_root)
    output_paths = {
        "per_model_metrics.csv": output / "per_model_metrics.csv",
        "per_training_seed_metrics.csv": (
            output / "per_training_seed_metrics.csv"
        ),
        "paired_accuracy_deltas.csv": output / "paired_accuracy_deltas.csv",
        "adaptive_assessment.csv": output / "adaptive_assessment.csv",
    }
    write_csv(
        output_paths["per_model_metrics.csv"],
        per_model_rows(validated.metrics),
    )
    write_csv(
        output_paths["per_training_seed_metrics.csv"],
        per_training_seed_rows(validated.metrics),
    )
    write_csv(
        output_paths["paired_accuracy_deltas.csv"],
        paired_accuracy_rows(validated.metrics),
    )
    write_csv(
        output_paths["adaptive_assessment.csv"],
        adaptive_assessment_rows(validated.metrics),
    )
    _recheck_input_hashes(validated)

    command = exact_command()
    revision = git_revision()
    source_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": EXPERIMENT_ID,
        "config_id": CONFIG_ID,
        "verification": "artifacts-verified",
        "evidence_label": "shot-simulation",
        "input_raw": {
            "raw_result_path": str(validated.result_path.parent),
            "result_path": str(validated.result_path),
            "result_sha256": validated.input_artifacts["result"]["sha256"],
            "artifacts": validated.input_artifacts,
        },
        "contract_validation": {
            "checkpoint_count": EXPECTED_CHECKPOINTS,
            "campaign_repeats_per_checkpoint": CAMPAIGN_REPEATS,
            "campaign_count": EXPECTED_CAMPAIGNS,
            "campaign_metric_row_count": EXPECTED_CAMPAIGN_ROWS,
            "eval_point_count": EXPECTED_EVAL_POINTS,
            "optimizer_runs": 0,
            "nfev": 0,
            "campaign_csv_json_jsonl_equal": True,
            "metrics_recomputed_equal": True,
            "all_checkpoint_hashes_verified": True,
            "eval_indices_hash_verified": True,
            "exact_probability_artifact_verified": True,
        },
        "per_model": validated.metrics["per_model"],
        "per_training_seed": validated.metrics["per_training_seed"],
        "paired_accuracy_deltas": validated.metrics[
            "paired_accuracy_deltas"
        ],
        "adaptive_thresholds": validated.metrics["adaptive_thresholds"],
        "adaptive_assessment": validated.metrics["adaptive_assessment"],
        "exact_checkpoint_metrics": validated.metrics[
            "exact_checkpoint_metrics"
        ],
        "output_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
        "command": command,
        "code_revision": revision,
        "summary_source": str(source_path),
        "summary_source_sha256": sha256_file(source_path),
    }
    summary["summary_content_sha256"] = canonical_json_hash(summary)
    summary_path = output / "summary.json"
    json_dump(summary_path, summary)
    checksums_path = write_sha256sums(
        output,
        (
            "per_model_metrics.csv",
            "per_training_seed_metrics.csv",
            "paired_accuracy_deltas.csv",
            "adaptive_assessment.csv",
            "summary.json",
        ),
    )
    print(
        "RESULT "
        + json.dumps(
            {
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "sha256sums": str(checksums_path),
                "sha256sums_sha256": sha256_file(checksums_path),
                "summary_content_sha256": summary[
                    "summary_content_sha256"
                ],
                "verification": summary["verification"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary_path


def summarize_m4(
    result_path: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    return write_m4_summary(
        validate_m4_raw(result_path),
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize one completed M4 raw run"
    )
    parser.add_argument(
        "--raw-result",
        type=Path,
        help=(
            "M4 result.json or its raw directory; defaults to the latest "
            "artifacts-verified run"
        ),
    )
    args = parser.parse_args()
    result_path = args.raw_result or latest_raw_result()
    summarize_m4(result_path)


if __name__ == "__main__":
    main()
