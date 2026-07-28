from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments import run_shot_evaluation as runner  # noqa: E402
from experiments import summarize_m4  # noqa: E402
from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    canonical_json_hash,
    sha256_file,
)


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _campaign_row(
    *,
    record: dict[str, Any],
    repeat_index: int,
    method_id: str,
) -> dict[str, Any]:
    fixed = method_id.startswith("fixed-")
    nominal_shots = (
        int(method_id.removeprefix("fixed-")) if fixed else None
    )
    if fixed:
        mean_shots = float(nominal_shots)
        stop_fractions = {
            shot: float(shot == nominal_shots)
            for shot in runner.DEFAULT_ADAPTIVE_CHECKPOINTS
        }
        total_shots = nominal_shots * 1000
        cap_hit = 0
        cap_hit_fraction = 0.0
        accuracy = 0.80 + 0.001 * (repeat_index % 2)
    else:
        stop_fractions = {128: 0.8, 512: 0.15, 2048: 0.05}
        mean_shots = sum(
            shot * fraction for shot, fraction in stop_fractions.items()
        )
        total_shots = int(mean_shots * 1000)
        cap_hit = 20
        cap_hit_fraction = 0.02
        accuracy = 0.82 + 0.001 * (repeat_index % 2)

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
        "rng_bit_generator": "PCG64",
        "rng_entropy": (
            f"2026:4:{record['model_code']}:{record['training_seed']}:"
            f"{repeat_index}"
        ),
        "rng_state_words": f"{repeat_index}:1:2:3",
        "method_id": method_id,
        "policy": "fixed" if fixed else "clopper-pearson-sequential",
        "nominal_shots": nominal_shots,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "exact_accuracy": 1.0,
        "n_points": 1000,
        "accuracy": accuracy,
        "exact_label_flip_rate": 1.0 - accuracy,
        "mean_shots": mean_shots,
        "median_shots": float(nominal_shots) if fixed else 128.0,
        "p90_shots": float(nominal_shots) if fixed else 512.0,
        "p95_shots": float(nominal_shots) if fixed else 2048.0,
        "stop_fraction_128": stop_fractions[128],
        "stop_fraction_512": stop_fractions[512],
        "stop_fraction_2048": stop_fractions[2048],
        "cap_hit": cap_hit,
        "cap_hit_fraction": cap_hit_fraction,
        "total_shots": total_shots,
        "alpha": None if fixed else 0.05,
        "cp_tail_probability": None if fixed else 0.05 / 6,
    }
    return row


def build_synthetic_raw(root: Path) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    dataset_hash = "d" * 64
    dataset_path = root / "dataset.npz"
    np.savez_compressed(dataset_path, marker=np.asarray(dataset_hash))

    records: list[dict[str, Any]] = []
    for spec in runner.MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            source_dir = root / "sources" / spec.model_id / f"seed-{seed}"
            source_dir.mkdir(parents=True)
            checkpoint = source_dir / "checkpoint.npz"
            np.savez_compressed(
                checkpoint,
                theta=np.zeros((1, spec.layers, 3)),
                alpha=np.zeros((1, spec.layers, 2)),
            )
            source_result = source_dir / "result.json"
            _dump_json(
                source_result,
                {
                    "experiment_id": spec.source_experiment_id,
                    "config_id": spec.source_config_id,
                    "seed": seed,
                },
            )
            records.append(
                {
                    "model_id": spec.model_id,
                    "model_code": spec.model_code,
                    "training_seed": seed,
                    "source_experiment_id": spec.source_experiment_id,
                    "source_config_id": spec.source_config_id,
                    "source_result": str(source_result),
                    "source_result_sha256": sha256_file(source_result),
                    "checkpoint": str(checkpoint),
                    "checkpoint_recorded_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "theta_shape": [1, spec.layers, 3],
                    "alpha_shape": [1, spec.layers, 2],
                    "dataset_hash": dataset_hash,
                }
            )
    records_path = raw_dir / "checkpoint_records.json"
    _dump_json(records_path, records)

    labels = np.asarray([0] * 500 + [1] * 500, dtype=np.int8)
    indices = np.arange(1000, dtype=np.int64)
    indices_payload = {
        "schema_version": 1,
        "purpose": "M4-fixed-adaptive-shots",
        "eval_seed": 2026,
        "sampling": "stratified-without-replacement",
        "per_class": 500,
        "point_count": 1000,
        "index_space": "test-set-relative",
        "test_relative_indices": indices.tolist(),
        "dataset_global_indices": (indices + 200).tolist(),
        "labels": labels.tolist(),
        "dataset_hash": dataset_hash,
    }
    indices_payload["indices_content_sha256"] = canonical_json_hash(
        indices_payload
    )
    indices_path = root / "eval_indices.json"
    _dump_json(indices_path, indices_payload)

    p0_row = np.where(labels == 0, 0.9, 0.1)
    exact_p0 = np.vstack([p0_row for _ in records])
    exact_path = raw_dir / "exact_probabilities.npz"
    np.savez_compressed(
        exact_path,
        exact_p0=exact_p0,
        exact_prediction=np.vstack(
            [runner.exact_predictions(row) for row in exact_p0]
        ),
        ground_truth_labels=labels,
        eval_test_relative_indices=indices,
        dataset_hash=np.asarray(dataset_hash),
        model_id=np.asarray([record["model_id"] for record in records]),
        training_seed=np.asarray(
            [record["training_seed"] for record in records],
            dtype=np.int64,
        ),
        checkpoint_sha256=np.asarray(
            [record["checkpoint_sha256"] for record in records]
        ),
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        for repeat_index in range(100):
            for method_id in summarize_m4.METHOD_ORDER:
                rows.append(
                    _campaign_row(
                        record=record,
                        repeat_index=repeat_index,
                        method_id=method_id,
                    )
                )
    json_path = raw_dir / "campaign_metrics.json"
    jsonl_path = raw_dir / "campaign_metrics.jsonl"
    csv_path = raw_dir / "campaign_metrics.csv"
    _dump_json(json_path, rows)
    with jsonl_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exact_metrics = [
        {
            "model_id": record["model_id"],
            "model_code": record["model_code"],
            "training_seed": record["training_seed"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "point_count": 1000,
            "exact_accuracy": 1.0,
            "max_probability_normalization_error": 0.0,
        }
        for record in records
    ]
    metrics = runner.aggregate_campaign_metrics(rows)
    metrics.update(
        {
            "experiment_id": "M4",
            "config_id": "fixed-adaptive-shots",
            "dataset_hash": dataset_hash,
            "eval_indices_file_sha256": sha256_file(indices_path),
            "exact_checkpoint_metrics": exact_metrics,
            "campaign_count": 1000,
            "campaign_metric_row_count": 4000,
        }
    )
    metrics_path = raw_dir / "metrics.json"
    _dump_json(metrics_path, metrics)

    environment_path = raw_dir / "environment.json"
    _dump_json(environment_path, {"python": sys.version})
    config_path = raw_dir / "config.json"
    config = {
        "schema_version": 1,
        "experiment_id": "M4",
        "config_id": "fixed-adaptive-shots",
        "evidence_label": "shot-simulation",
        "run_kind": "frozen-checkpoint-shot-evaluation",
        "optimizer_runs": 0,
        "dataset_hash": dataset_hash,
        "dataset_artifact": str(dataset_path),
        "eval_seed": 2026,
        "sampling": "stratified-without-replacement",
        "points_per_class": 500,
        "eval_point_count": 1000,
        "fixed_shots": [128, 512, 2048],
        "adaptive_checkpoints": [128, 512, 2048],
        "adaptive_alpha": 0.05,
        "cp_tail_probability": 0.05 / 6,
        "campaign_repeats_per_checkpoint": 100,
        "checkpoint_count": 10,
        "eval_indices": str(indices_path),
        "checkpoint_records": str(records_path),
        "command": "synthetic-test",
        "code_revision": {"head": "test"},
    }
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
        }
    }
    config["config_fingerprint"] = canonical_json_hash(fingerprint_payload)
    _dump_json(config_path, config)

    result = {
        "schema_version": 1,
        "experiment_id": "M4",
        "config_id": "fixed-adaptive-shots",
        "evidence_label": "shot-simulation",
        "verification": "artifacts-verified",
        "optimizer_runs": 0,
        "nfev": 0,
        "dataset_hash": dataset_hash,
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "checkpoint_count": 10,
        "campaign_repeats_per_checkpoint": 100,
        "campaign_count": 1000,
        "campaign_metric_row_count": 4000,
        "eval_indices": str(indices_path),
        "eval_indices_file_sha256": sha256_file(indices_path),
        "eval_indices_content_sha256": indices_payload[
            "indices_content_sha256"
        ],
        "checkpoint_records": str(records_path),
        "checkpoint_records_sha256": sha256_file(records_path),
        "exact_probabilities": str(exact_path),
        "exact_probabilities_sha256": sha256_file(exact_path),
        "campaign_metrics_json": str(json_path),
        "campaign_metrics_json_sha256": sha256_file(json_path),
        "campaign_metrics_jsonl": str(jsonl_path),
        "campaign_metrics_jsonl_sha256": sha256_file(jsonl_path),
        "campaign_metrics_csv": str(csv_path),
        "campaign_metrics_csv_sha256": sha256_file(csv_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "environment": str(environment_path),
        "environment_sha256": sha256_file(environment_path),
        "raw_result_path": str(raw_dir),
    }
    result_path = raw_dir / "result.json"
    _dump_json(result_path, result)
    return result_path


def test_strict_validation_accepts_complete_consistent_raw(
    tmp_path: Path,
) -> None:
    result_path = build_synthetic_raw(tmp_path)

    validated = summarize_m4.validate_m4_raw(result_path)

    assert len(validated.checkpoint_records) == 10
    assert len(validated.campaign_rows) == 4000
    assert len(validated.metrics["per_model"]) == 8
    assert len(validated.metrics["per_training_seed"]) == 40
    assert len(validated.metrics["paired_accuracy_deltas"]) == 6
    assert validated.result["optimizer_runs"] == 0
    assert validated.result["nfev"] == 0


def test_summary_writes_requested_csvs_and_hash_manifest(
    tmp_path: Path,
) -> None:
    result_path = build_synthetic_raw(tmp_path / "input")
    validated = summarize_m4.validate_m4_raw(result_path)

    summary_path = summarize_m4.write_m4_summary(
        validated,
        output_root=tmp_path / "summary-root",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_csv_lines = {
        "per_model_metrics.csv": 9,
        "per_training_seed_metrics.csv": 41,
        "paired_accuracy_deltas.csv": 31,
        "adaptive_assessment.csv": 13,
    }
    assert summary_path.parent.parent == tmp_path / "summary-root"
    for name, expected_lines in expected_csv_lines.items():
        path = summary_path.parent / name
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8").splitlines()) == expected_lines
        assert summary["output_files"][name]["sha256"] == sha256_file(path)
    checksums_path = summary_path.parent / "SHA256SUMS"
    checksum_lines = checksums_path.read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 5
    expected_checksum_names = [
        "per_model_metrics.csv",
        "per_training_seed_metrics.csv",
        "paired_accuracy_deltas.csv",
        "adaptive_assessment.csv",
        "summary.json",
    ]
    for line, name in zip(checksum_lines, expected_checksum_names):
        digest, filename = line.split("  ", maxsplit=1)
        assert filename == name
        assert digest == sha256_file(summary_path.parent / name)
    assert summary["verification"] == "artifacts-verified"
    assert summary["contract_validation"]["campaign_count"] == 1000
    assert summary["contract_validation"]["optimizer_runs"] == 0
    content_hash = summary.pop("summary_content_sha256")
    assert canonical_json_hash(summary) == content_hash


def test_sha256sums_writer_is_exclusive(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")

    summarize_m4.write_sha256sums(tmp_path, ("summary.json",))

    with pytest.raises(FileExistsError):
        summarize_m4.write_sha256sums(tmp_path, ("summary.json",))


def test_validation_rejects_cross_format_campaign_disagreement(
    tmp_path: Path,
) -> None:
    result_path = build_synthetic_raw(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    json_path = Path(result["campaign_metrics_json"])
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    rows[0]["accuracy"] = 0.123
    _dump_json(json_path, rows)
    result["campaign_metrics_json_sha256"] = sha256_file(json_path)
    _dump_json(result_path, result)

    with pytest.raises(ValueError, match="JSON and JSONL rows differ"):
        summarize_m4.validate_m4_raw(result_path)


def test_validation_rejects_checkpoint_mutation(tmp_path: Path) -> None:
    result_path = build_synthetic_raw(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = json.loads(
        Path(result["checkpoint_records"]).read_text(encoding="utf-8")
    )
    checkpoint = Path(records[0]["checkpoint"])
    with checkpoint.open("ab") as handle:
        handle.write(b"mutation")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        summarize_m4.validate_m4_raw(result_path)
