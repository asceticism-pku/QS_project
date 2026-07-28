#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
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

from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    controlled_initialization,
    git_revision,
    json_dump,
    load_checkpoint,
    sha256_file,
)

PARITY_TOLERANCE = 1e-10
ACCURACY_DROP_TOLERANCE = 0.005


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    config_id: str
    source_stage: str
    layers: int
    parameter_count: int


MODEL_SPECS = (
    ModelSpec("l4-base", "1q-l4-paper_squared", "M1", 4, 20),
    ModelSpec("l4-to-l3-pruned", "l4-to-l3-pruned", "M3", 3, 15),
    ModelSpec("l4-truncate-last", "l4-truncate-last", "M3", 3, 15),
    ModelSpec("l3-scratch", "l3-scratch", "M3", 3, 15),
)
MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def referenced_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_keys(payload: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def mean_sample_sd(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def unique_summary_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ROOT / "results" / "summary" / "M3" / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def latest_compile_result() -> Path:
    root = ROOT / "results" / "raw" / "M3" / "compile-audit" / "seed-2026"
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise ValueError(f"missing M3 compile audit under {root}")
    return max(paths, key=lambda path: path.parent.name)


def checked_referenced_artifact(
    payload: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    label: str,
) -> Path:
    require_keys(payload, (path_key, sha_key), label)
    path = referenced_path(str(payload[path_key]))
    if not path.is_file():
        raise ValueError(f"{label} artifact does not exist: {path}")
    actual = sha256_file(path)
    expected = str(payload[sha_key])
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch for {path}: {actual} != {expected}"
        )
    return path


def load_compile_inputs() -> tuple[
    Path,
    dict[str, Any],
    Path,
    list[dict[str, Any]],
    Path,
    list[dict[str, Any]],
]:
    result_path = latest_compile_result()
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError(f"compile result is not an object: {result_path}")
    require_keys(
        result,
        (
            "experiment_id",
            "config_id",
            "compile_seed",
            "verification",
            "all_probability_parity",
            "all_label_parity",
            "max_probability_error",
            "summary",
            "summary_sha256",
            "checkpoint_records",
            "checkpoint_records_sha256",
        ),
        "M3 compile result",
    )
    if result["experiment_id"] != "M3" or result["config_id"] != "compile-audit":
        raise ValueError(f"unexpected compile result identity: {result_path}")
    if int(result["compile_seed"]) != 2026:
        raise ValueError(f"unexpected compile seed in {result_path}")

    summary_path = checked_referenced_artifact(
        result,
        "summary",
        "summary_sha256",
        label="M3 compile summary",
    )
    records_path = checked_referenced_artifact(
        result,
        "checkpoint_records",
        "checkpoint_records_sha256",
        label="M3 compile checkpoint records",
    )
    summary_rows = read_json(summary_path)
    checkpoint_records = read_json(records_path)
    if not isinstance(summary_rows, list):
        raise ValueError(f"compile summary is not a list: {summary_path}")
    if not isinstance(checkpoint_records, list):
        raise ValueError(f"checkpoint records are not a list: {records_path}")
    return (
        result_path,
        result,
        summary_path,
        summary_rows,
        records_path,
        checkpoint_records,
    )


def load_training_inputs(
    checkpoint_records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, Path]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, Path]],
]:
    record_map: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(checkpoint_records):
        if not isinstance(record, dict):
            raise ValueError(f"checkpoint record {index} is not an object")
        require_keys(
            record,
            ("config_id", "seed", "checkpoint", "checkpoint_sha256", "source_result"),
            f"checkpoint record {index}",
        )
        key = (str(record["config_id"]), int(record["seed"]))
        if key in record_map:
            raise ValueError(f"duplicate compile checkpoint record: {key}")
        record_map[key] = record

    expected_keys = {
        (spec.model_id, seed)
        for spec in MODEL_SPECS
        for seed in CONTROLLED_SEEDS
    }
    missing = sorted(expected_keys - set(record_map))
    if missing:
        raise ValueError(f"M3 compile audit missing checkpoint records: {missing}")

    results: dict[str, dict[int, dict[str, Any]]] = {
        spec.model_id: {} for spec in MODEL_SPECS
    }
    result_paths: dict[str, dict[int, Path]] = {
        spec.model_id: {} for spec in MODEL_SPECS
    }
    configs: dict[str, dict[int, dict[str, Any]]] = {
        spec.model_id: {} for spec in MODEL_SPECS
    }
    config_paths: dict[str, dict[int, Path]] = {
        spec.model_id: {} for spec in MODEL_SPECS
    }

    for spec in MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            record = record_map[(spec.model_id, seed)]
            if (
                record.get("model_id") != spec.model_id
                or record.get("source_config_id") != spec.config_id
                or record.get("source_stage") != spec.source_stage
            ):
                raise ValueError(
                    f"compile checkpoint identity mismatch for "
                    f"{spec.model_id}, seed={seed}"
                )
            result_path = referenced_path(str(record["source_result"]))
            if not result_path.is_file():
                raise ValueError(f"training result does not exist: {result_path}")
            result = read_json(result_path)
            if not isinstance(result, dict):
                raise ValueError(f"training result is not an object: {result_path}")
            require_keys(
                result,
                (
                    "experiment_id",
                    "config_id",
                    "seed",
                    "dataset_hash",
                    "loss_id",
                    "verification",
                    "optimizer",
                    "train_metrics",
                    "test_metrics",
                    "checkpoint",
                    "checkpoint_sha256",
                    "initial_checkpoint",
                    "initial_checkpoint_sha256",
                ),
                f"training result {result_path}",
            )
            identity = (
                str(result["experiment_id"]),
                str(result["config_id"]),
                int(result["seed"]),
            )
            expected_identity = (spec.source_stage, spec.config_id, seed)
            if identity != expected_identity:
                raise ValueError(
                    f"training result identity {identity} != {expected_identity}: "
                    f"{result_path}"
                )
            if result["loss_id"] != "paper_squared":
                raise ValueError(f"unexpected loss in {result_path}")
            if result["verification"] != "artifacts-verified":
                raise ValueError(
                    f"training artifact is not verified: {result_path} "
                    f"({result['verification']})"
                )

            checkpoint_path = referenced_path(str(result["checkpoint"]))
            if not checkpoint_path.is_file():
                raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
            checkpoint_sha256 = sha256_file(checkpoint_path)
            if checkpoint_sha256 != str(result["checkpoint_sha256"]):
                raise ValueError(
                    f"training checkpoint hash mismatch: {checkpoint_path}"
                )
            if checkpoint_path.resolve() != referenced_path(
                str(record["checkpoint"])
            ).resolve():
                raise ValueError(
                    f"compile/training checkpoint path mismatch for "
                    f"{spec.config_id}, seed={seed}"
                )
            if checkpoint_sha256 != str(record["checkpoint_sha256"]):
                raise ValueError(
                    f"compile/training checkpoint hash mismatch for "
                    f"{spec.config_id}, seed={seed}"
                )
            initial_checkpoint_path = referenced_path(
                str(result["initial_checkpoint"])
            )
            if not initial_checkpoint_path.is_file():
                raise ValueError(
                    f"initial checkpoint does not exist: {initial_checkpoint_path}"
                )
            if sha256_file(initial_checkpoint_path) != str(
                result["initial_checkpoint_sha256"]
            ):
                raise ValueError(
                    f"initial checkpoint hash mismatch: {initial_checkpoint_path}"
                )

            config_path = result_path.parent / "config.json"
            if not config_path.is_file():
                raise ValueError(f"missing training config: {config_path}")
            config = read_json(config_path)
            if not isinstance(config, dict):
                raise ValueError(f"training config is not an object: {config_path}")
            require_keys(
                config,
                (
                    "experiment_id",
                    "config_id",
                    "init_seed",
                    "layers",
                    "parameter_count",
                    "parent_checkpoint",
                    "removed_layer",
                ),
                f"training config {config_path}",
            )
            if (
                config["experiment_id"] != spec.source_stage
                or config["config_id"] != spec.config_id
                or int(config["init_seed"]) != seed
                or int(config["layers"]) != spec.layers
                or int(config["parameter_count"]) != spec.parameter_count
            ):
                raise ValueError(f"training config contract mismatch: {config_path}")

            theta, alpha, weights = load_checkpoint(checkpoint_path)
            if weights is not None:
                raise ValueError(
                    f"M3 ordinary checkpoint has weights: {checkpoint_path}"
                )
            if theta.shape != (1, spec.layers, 3) or alpha.shape != (
                1,
                spec.layers,
                2,
            ):
                raise ValueError(
                    f"checkpoint shape mismatch for {spec.model_id}, seed={seed}: "
                    f"theta={theta.shape}, alpha={alpha.shape}"
                )

            results[spec.model_id][seed] = result
            result_paths[spec.model_id][seed] = result_path
            configs[spec.model_id][seed] = config
            config_paths[spec.model_id][seed] = config_path

    return results, result_paths, configs, config_paths


def latest_selection_result(
    seed: int,
    *,
    base_result_path: Path,
    base_checkpoint_path: Path,
    base_checkpoint_sha256: str,
) -> Path:
    root = (
        ROOT
        / "results"
        / "raw"
        / "M3"
        / "pruning-selection"
        / f"seed-{seed}"
    )
    matches: list[Path] = []
    for path in sorted(root.glob("*/result.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("experiment_id") == "M3"
            and payload.get("config_id") == "pruning-selection"
            and payload.get("seed") == seed
            and payload.get("verification") == "artifacts-verified"
            and referenced_path(str(payload.get("base_result", ""))).resolve()
            == base_result_path.resolve()
            and referenced_path(str(payload.get("base_checkpoint", ""))).resolve()
            == base_checkpoint_path.resolve()
            and payload.get("base_checkpoint_sha256") == base_checkpoint_sha256
        ):
            matches.append(path)
    if not matches:
        raise ValueError(
            f"missing verified pruning selection for seed={seed} and base "
            f"{base_checkpoint_path}"
        )
    return max(matches, key=lambda path: path.parent.name)


def load_selection_inputs(
    results: dict[str, dict[int, dict[str, Any]]],
    result_paths: dict[str, dict[int, Path]],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, Path],
    list[dict[str, Any]],
]:
    selections: dict[int, dict[str, Any]] = {}
    selection_paths: dict[int, Path] = {}
    rows: list[dict[str, Any]] = []
    for seed in CONTROLLED_SEEDS:
        base = results["l4-base"][seed]
        base_result_path = result_paths["l4-base"][seed]
        base_checkpoint = referenced_path(str(base["checkpoint"]))
        selection_path = latest_selection_result(
            seed,
            base_result_path=base_result_path,
            base_checkpoint_path=base_checkpoint,
            base_checkpoint_sha256=str(base["checkpoint_sha256"]),
        )
        selection = read_json(selection_path)
        if not isinstance(selection, dict):
            raise ValueError(f"selection result is not an object: {selection_path}")
        require_keys(
            selection,
            (
                "dataset_hash",
                "base_train_loss_recomputed",
                "candidates",
                "selected_layer_zero_based",
                "selected_layer_one_based",
                "selection_rule",
                "verification",
            ),
            f"selection result {selection_path}",
        )
        if selection["dataset_hash"] != base["dataset_hash"]:
            raise ValueError(f"selection/base dataset mismatch: {selection_path}")
        candidates = selection["candidates"]
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(
                f"selection must contain four layer candidates: {selection_path}"
            )
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"invalid pruning candidate: {selection_path}")
            require_keys(
                candidate,
                (
                    "layer_zero_based",
                    "layer_one_based",
                    "unfinetuned_train_loss",
                    "loss_increase_from_base",
                ),
                f"selection candidate {selection_path}",
            )
            normalized_candidates.append(candidate)
        layer_indices = sorted(
            int(candidate["layer_zero_based"])
            for candidate in normalized_candidates
        )
        if layer_indices != [0, 1, 2, 3]:
            raise ValueError(
                f"selection candidates have wrong layer indices: {selection_path}"
            )
        expected = min(
            normalized_candidates,
            key=lambda candidate: (
                float(candidate["loss_increase_from_base"]),
                int(candidate["layer_zero_based"]),
            ),
        )
        selected_zero = int(selection["selected_layer_zero_based"])
        selected_one = int(selection["selected_layer_one_based"])
        selection_rule_verified = (
            selected_zero == int(expected["layer_zero_based"])
            and selected_one == selected_zero + 1
        )
        base_loss_matches = math.isclose(
            float(selection["base_train_loss_recomputed"]),
            float(base["optimizer"]["fun"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        selected_candidate = next(
            candidate
            for candidate in normalized_candidates
            if int(candidate["layer_zero_based"]) == selected_zero
        )
        row: dict[str, Any] = {
            "seed": seed,
            "selected_layer_zero_based": selected_zero,
            "selected_layer_one_based": selected_one,
            "base_train_loss_recomputed": selection[
                "base_train_loss_recomputed"
            ],
            "base_final_loss_recorded": base["optimizer"]["fun"],
            "base_loss_matches": base_loss_matches,
            "selected_unfinetuned_train_loss": selected_candidate[
                "unfinetuned_train_loss"
            ],
            "selected_loss_increase_from_base": selected_candidate[
                "loss_increase_from_base"
            ],
            "selection_rule_verified": selection_rule_verified,
            "verification": selection["verification"],
            "source_result_path": str(selection_path),
            "source_result_sha256": sha256_file(selection_path),
        }
        for candidate in normalized_candidates:
            layer = int(candidate["layer_zero_based"])
            row[f"layer_{layer}_unfinetuned_train_loss"] = candidate[
                "unfinetuned_train_loss"
            ]
            row[f"layer_{layer}_loss_increase_from_base"] = candidate[
                "loss_increase_from_base"
            ]
        selections[seed] = selection
        selection_paths[seed] = selection_path
        rows.append(row)
    return selections, selection_paths, rows


def initialization_contract(
    model_id: str,
    seed: int,
    *,
    result: dict[str, Any],
    config: dict[str, Any],
    base: dict[str, Any],
    selected_layer: int,
) -> bool:
    initial_path = referenced_path(str(result["initial_checkpoint"]))
    if not initial_path.is_file():
        return False
    initial_theta, initial_alpha, initial_weights = load_checkpoint(initial_path)
    if initial_weights is not None:
        return False
    if model_id == "l4-base":
        expected_theta, expected_alpha = controlled_initialization(1, 4, seed)
        return (
            config["parent_checkpoint"] is None
            and config["removed_layer"] is None
            and np.array_equal(initial_theta, expected_theta)
            and np.array_equal(initial_alpha, expected_alpha)
        )
    if model_id == "l3-scratch":
        expected_theta, expected_alpha = controlled_initialization(1, 3, seed)
        return (
            config["parent_checkpoint"] is None
            and config["removed_layer"] is None
            and np.array_equal(initial_theta, expected_theta)
            and np.array_equal(initial_alpha, expected_alpha)
        )

    base_checkpoint = referenced_path(str(base["checkpoint"]))
    base_theta, base_alpha, base_weights = load_checkpoint(base_checkpoint)
    if base_weights is not None:
        return False
    removed_layer = selected_layer if model_id == "l4-to-l3-pruned" else 3
    return (
        referenced_path(str(config["parent_checkpoint"])).resolve()
        == base_checkpoint.resolve()
        and int(config["removed_layer"]) == removed_layer
        and np.array_equal(initial_theta, np.delete(base_theta, removed_layer, axis=1))
        and np.array_equal(initial_alpha, np.delete(base_alpha, removed_layer, axis=1))
    )


def training_rows_and_summary(
    results: dict[str, dict[int, dict[str, Any]]],
    result_paths: dict[str, dict[int, Path]],
    configs: dict[str, dict[int, dict[str, Any]]],
    selections: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_summary: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            result = results[spec.model_id][seed]
            config = configs[spec.model_id][seed]
            base = results["l4-base"][seed]
            optimizer_nfev = int(result["optimizer"]["nfev"])
            base_nfev = int(base["optimizer"]["nfev"])
            pipeline_nfev = (
                base_nfev + optimizer_nfev
                if spec.model_id
                in {"l4-to-l3-pruned", "l4-truncate-last"}
                else optimizer_nfev
            )
            selected_layer = int(
                selections[seed]["selected_layer_zero_based"]
            )
            initialization_verified = initialization_contract(
                spec.model_id,
                seed,
                result=result,
                config=config,
                base=base,
                selected_layer=selected_layer,
            )
            rows.append(
                {
                    "model_id": spec.model_id,
                    "source_config_id": spec.config_id,
                    "source_stage": spec.source_stage,
                    "seed": seed,
                    "layers": spec.layers,
                    "parameter_count": spec.parameter_count,
                    "layer_reduction_fraction_vs_base": (
                        (4 - spec.layers) / 4
                    ),
                    "parameter_reduction_fraction_vs_base": (
                        (20 - spec.parameter_count) / 20
                    ),
                    "removed_layer_zero_based": config["removed_layer"],
                    "selected_layer_zero_based": (
                        selected_layer
                        if spec.model_id == "l4-to-l3-pruned"
                        else None
                    ),
                    "train_accuracy": result["train_metrics"]["accuracy"],
                    "test_accuracy": result["test_metrics"]["accuracy"],
                    "train_mean_true_margin": result["train_metrics"][
                        "mean_true_margin"
                    ],
                    "test_mean_true_margin": result["test_metrics"][
                        "mean_true_margin"
                    ],
                    "final_loss": result["optimizer"]["fun"],
                    "optimizer_success": result["optimizer"]["success"],
                    "optimizer_status": result["optimizer"]["status"],
                    "optimizer_message": result["optimizer"]["message"],
                    "optimizer_nfev": optimizer_nfev,
                    "base_nfev": base_nfev,
                    "pipeline_nfev": pipeline_nfev,
                    "pipeline_definition": (
                        "base_nfev + finetune_nfev"
                        if spec.model_id
                        in {"l4-to-l3-pruned", "l4-truncate-last"}
                        else "model_optimizer_nfev"
                    ),
                    "initialization_contract_verified": initialization_verified,
                    "checkpoint": result["checkpoint"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "raw_result_path": str(result_paths[spec.model_id][seed]),
                    "raw_result_sha256": sha256_file(
                        result_paths[spec.model_id][seed]
                    ),
                }
            )

        model_rows = [row for row in rows if row["model_id"] == spec.model_id]
        model_summary[spec.model_id] = {
            "source_config_id": spec.config_id,
            "source_stage": spec.source_stage,
            "layers": spec.layers,
            "parameter_count": spec.parameter_count,
            "train_accuracy": mean_sample_sd(
                [float(row["train_accuracy"]) for row in model_rows]
            ),
            "test_accuracy": mean_sample_sd(
                [float(row["test_accuracy"]) for row in model_rows]
            ),
            "final_loss": mean_sample_sd(
                [float(row["final_loss"]) for row in model_rows]
            ),
            "optimizer_nfev": mean_sample_sd(
                [float(row["optimizer_nfev"]) for row in model_rows]
            ),
            "pipeline_nfev": mean_sample_sd(
                [float(row["pipeline_nfev"]) for row in model_rows]
            ),
            "optimizer_success_count": sum(
                bool(row["optimizer_success"]) for row in model_rows
            ),
            "all_initialization_contracts_verified": all(
                bool(row["initialization_contract_verified"])
                for row in model_rows
            ),
        }
    return rows, model_summary


def paired_rows_and_summary(
    results: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in CONTROLLED_SEEDS:
        base = results["l4-base"][seed]
        pruned = results["l4-to-l3-pruned"][seed]
        truncate = results["l4-truncate-last"][seed]
        base_nfev = int(base["optimizer"]["nfev"])
        pruned_nfev = int(pruned["optimizer"]["nfev"])
        truncate_nfev = int(truncate["optimizer"]["nfev"])
        row = {
            "seed": seed,
            "base_test_accuracy": base["test_metrics"]["accuracy"],
            "pruned_test_accuracy": pruned["test_metrics"]["accuracy"],
            "truncate_test_accuracy": truncate["test_metrics"]["accuracy"],
            "pruned_minus_base_test_accuracy": (
                pruned["test_metrics"]["accuracy"]
                - base["test_metrics"]["accuracy"]
            ),
            "truncate_minus_base_test_accuracy": (
                truncate["test_metrics"]["accuracy"]
                - base["test_metrics"]["accuracy"]
            ),
            "pruned_minus_truncate_test_accuracy": (
                pruned["test_metrics"]["accuracy"]
                - truncate["test_metrics"]["accuracy"]
            ),
            "pruned_minus_base_train_accuracy": (
                pruned["train_metrics"]["accuracy"]
                - base["train_metrics"]["accuracy"]
            ),
            "truncate_minus_base_train_accuracy": (
                truncate["train_metrics"]["accuracy"]
                - base["train_metrics"]["accuracy"]
            ),
            "pruned_minus_truncate_train_accuracy": (
                pruned["train_metrics"]["accuracy"]
                - truncate["train_metrics"]["accuracy"]
            ),
            "pruned_minus_base_final_loss": (
                pruned["optimizer"]["fun"] - base["optimizer"]["fun"]
            ),
            "truncate_minus_base_final_loss": (
                truncate["optimizer"]["fun"] - base["optimizer"]["fun"]
            ),
            "pruned_minus_truncate_final_loss": (
                pruned["optimizer"]["fun"] - truncate["optimizer"]["fun"]
            ),
            "base_optimizer_nfev": base_nfev,
            "pruned_finetune_nfev": pruned_nfev,
            "truncate_finetune_nfev": truncate_nfev,
            "pruned_pipeline_nfev": base_nfev + pruned_nfev,
            "truncate_pipeline_nfev": base_nfev + truncate_nfev,
            "pruned_minus_base_pipeline_nfev": pruned_nfev,
            "truncate_minus_base_pipeline_nfev": truncate_nfev,
            "pruned_minus_truncate_pipeline_nfev": (
                pruned_nfev - truncate_nfev
            ),
        }
        rows.append(row)

    directions = (
        "pruned_minus_base",
        "truncate_minus_base",
        "pruned_minus_truncate",
    )
    metrics = (
        "test_accuracy",
        "train_accuracy",
        "final_loss",
        "pipeline_nfev",
    )
    summary = {
        direction: {
            metric: mean_sample_sd(
                [
                    float(row[f"{direction}_{metric}"])
                    for row in rows
                ]
            )
            for metric in metrics
        }
        for direction in directions
    }
    return rows, summary


def compile_rows_and_summary(
    compile_result: dict[str, Any],
    raw_summary_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    raw_map: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, row in enumerate(raw_summary_rows):
        if not isinstance(row, dict):
            raise ValueError(f"compile summary row {index} is not an object")
        require_keys(
            row,
            (
                "config_id",
                "seed",
                "optimization_level",
                "n_points",
                "median_target_depth",
                "median_rz",
                "median_sx",
                "median_x",
                "median_cz",
                "max_probability_error",
                "all_probability_parity",
                "all_label_parity",
            ),
            f"compile summary row {index}",
        )
        model_id = str(row["config_id"])
        if model_id not in MODEL_BY_ID:
            continue
        spec = MODEL_BY_ID[model_id]
        if (
            row.get("model_id") != spec.model_id
            or row.get("source_config_id") != spec.config_id
            or row.get("source_stage") != spec.source_stage
        ):
            raise ValueError(
                f"compile summary identity mismatch for row {index}"
            )
        key = (
            model_id,
            int(row["seed"]),
            int(row["optimization_level"]),
        )
        if key in raw_map:
            raise ValueError(f"duplicate compile summary row: {key}")
        raw_map[key] = row

    expected_keys = {
        (spec.model_id, seed, level)
        for spec in MODEL_SPECS
        for seed in CONTROLLED_SEEDS
        for level in (0, 3)
    }
    missing = sorted(expected_keys - set(raw_map))
    if missing:
        raise ValueError(f"M3 compile summary missing rows: {missing}")

    resource_rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            for level in (0, 3):
                raw = raw_map[(spec.model_id, seed, level)]
                resource_rows.append(
                    {
                        "model_id": spec.model_id,
                        "source_config_id": spec.config_id,
                        "seed": seed,
                        "optimization_level": level,
                        "n_points": raw["n_points"],
                        "median_target_depth": raw["median_target_depth"],
                        "median_rz": raw["median_rz"],
                        "median_sx": raw["median_sx"],
                        "median_x": raw["median_x"],
                        "median_cz": raw["median_cz"],
                        "max_probability_error": raw[
                            "max_probability_error"
                        ],
                        "all_probability_parity": raw[
                            "all_probability_parity"
                        ],
                        "all_label_parity": raw["all_label_parity"],
                    }
                )

    depth_rows: list[dict[str, Any]] = []
    for seed in CONTROLLED_SEEDS:
        for level in (0, 3):
            values = {
                spec.model_id: float(
                    raw_map[(spec.model_id, seed, level)][
                        "median_target_depth"
                    ]
                )
                for spec in MODEL_SPECS
            }
            depth_rows.append(
                {
                    "seed": seed,
                    "optimization_level": level,
                    "base_median_target_depth": values["l4-base"],
                    "pruned_median_target_depth": values[
                        "l4-to-l3-pruned"
                    ],
                    "truncate_median_target_depth": values[
                        "l4-truncate-last"
                    ],
                    "scratch_median_target_depth": values["l3-scratch"],
                    "pruned_minus_base_depth": (
                        values["l4-to-l3-pruned"] - values["l4-base"]
                    ),
                    "truncate_minus_base_depth": (
                        values["l4-truncate-last"] - values["l4-base"]
                    ),
                    "scratch_minus_base_depth": (
                        values["l3-scratch"] - values["l4-base"]
                    ),
                    "pruned_nonincrease_vs_base": (
                        values["l4-to-l3-pruned"] <= values["l4-base"]
                    ),
                    "pruned_strict_decrease_vs_base": (
                        values["l4-to-l3-pruned"] < values["l4-base"]
                    ),
                }
            )

    level_zero_rows = [
        row for row in depth_rows if row["optimization_level"] == 0
    ]
    level_three_rows = [
        row for row in depth_rows if row["optimization_level"] == 3
    ]
    level_zero_nonincrease = all(
        bool(row["pruned_nonincrease_vs_base"]) for row in level_zero_rows
    )
    level_zero_strict_count = sum(
        bool(row["pruned_strict_decrease_vs_base"])
        for row in level_zero_rows
    )

    level_three_changes: dict[str, Any] = {}
    for name in ("pruned", "truncate", "scratch"):
        deltas = [
            float(row[f"{name}_minus_base_depth"])
            for row in level_three_rows
        ]
        level_three_changes[f"{name}_minus_base"] = {
            **mean_sample_sd(deltas),
            "per_seed": {
                str(row["seed"]): float(
                    row[f"{name}_minus_base_depth"]
                )
                for row in level_three_rows
            },
            "all_nonincrease": all(delta <= 0.0 for delta in deltas),
            "strict_decrease_count": sum(delta < 0.0 for delta in deltas),
            "any_strict_decrease": any(delta < 0.0 for delta in deltas),
        }

    row_probability_parity = all(
        bool(row["all_probability_parity"]) for row in resource_rows
    )
    row_label_parity = all(
        bool(row["all_label_parity"]) for row in resource_rows
    )
    row_max_probability_error = max(
        float(row["max_probability_error"]) for row in resource_rows
    )
    all_rows_have_100_points = all(
        int(row["n_points"]) == 100 for row in resource_rows
    )
    expected_checkpoint_count = len(MODEL_SPECS) * len(CONTROLLED_SEEDS)
    expected_raw_row_count = expected_checkpoint_count * 100 * 2
    compile_counts_complete = (
        int(compile_result.get("checkpoint_count", -1))
        == expected_checkpoint_count
        and int(compile_result.get("point_count_per_checkpoint", -1)) == 100
        and int(compile_result.get("row_count", -1))
        == expected_raw_row_count
        and all_rows_have_100_points
    )
    probability_result_consistent = (
        bool(compile_result["all_probability_parity"])
        == row_probability_parity
        and math.isclose(
            float(compile_result["max_probability_error"]),
            row_max_probability_error,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
    )
    label_result_consistent = (
        bool(compile_result["all_label_parity"]) == row_label_parity
    )
    parity = {
        "tolerance_strictly_less_than": PARITY_TOLERANCE,
        "max_probability_error": row_max_probability_error,
        "all_probability_parity_from_rows": row_probability_parity,
        "all_label_parity_from_rows": row_label_parity,
        "all_probability_parity_from_result": bool(
            compile_result["all_probability_parity"]
        ),
        "all_label_parity_from_result": bool(
            compile_result["all_label_parity"]
        ),
        "probability_result_consistent_with_rows": probability_result_consistent,
        "label_result_consistent_with_rows": label_result_consistent,
        "probability_condition_passed": (
            row_probability_parity
            and bool(compile_result["all_probability_parity"])
            and probability_result_consistent
            and row_max_probability_error < PARITY_TOLERANCE
        ),
        "label_condition_passed": (
            row_label_parity
            and bool(compile_result["all_label_parity"])
            and label_result_consistent
        ),
        "comparison_scope": (
            "logical versus compiled outputs for the same checkpoint; "
            "not L4 versus L3 equivalence"
        ),
    }
    compile_summary = {
        "verification": compile_result["verification"],
        "compile_seed": compile_result["compile_seed"],
        "seed_transpiler": compile_result.get("seed_transpiler"),
        "checkpoint_count": compile_result.get("checkpoint_count"),
        "point_count_per_checkpoint": compile_result.get(
            "point_count_per_checkpoint"
        ),
        "row_count": compile_result.get("row_count"),
        "completeness": {
            "expected_checkpoint_count": expected_checkpoint_count,
            "expected_point_count_per_checkpoint": 100,
            "expected_raw_row_count": expected_raw_row_count,
            "all_summary_rows_have_100_points": all_rows_have_100_points,
            "condition_passed": compile_counts_complete,
        },
        "level0_pruned_vs_base": {
            "per_seed": {
                str(row["seed"]): {
                    "base_median_target_depth": row[
                        "base_median_target_depth"
                    ],
                    "pruned_median_target_depth": row[
                        "pruned_median_target_depth"
                    ],
                    "delta_pruned_minus_base": row[
                        "pruned_minus_base_depth"
                    ],
                    "nonincrease": row["pruned_nonincrease_vs_base"],
                    "strict_decrease": row[
                        "pruned_strict_decrease_vs_base"
                    ],
                }
                for row in level_zero_rows
            },
            "all_five_nonincrease": level_zero_nonincrease,
            "strict_decrease_count": level_zero_strict_count,
            "requires_at_least_four_strict": True,
            "condition_passed": (
                level_zero_nonincrease and level_zero_strict_count >= 4
            ),
        },
        "level3_native_depth_changes": level_three_changes,
        "level3_native_depth_interpretation": (
            "A layer/parameter reduction with unchanged level-3 target depth "
            "is model/template compression, not demonstrated native-gate "
            "depth reduction."
        ),
        "parity": parity,
    }
    return resource_rows, depth_rows, compile_summary


def source_artifact_rows(
    *,
    results: dict[str, dict[int, dict[str, Any]]],
    result_paths: dict[str, dict[int, Path]],
    config_paths: dict[str, dict[int, Path]],
    selection_paths: dict[int, Path],
    compile_result_path: Path,
    compile_result: dict[str, Any],
    compile_summary_path: Path,
    compile_records_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        kind: str,
        path: Path,
        *,
        model_id: str = "",
        seed: int | str = "",
    ) -> None:
        rows.append(
            {
                "kind": kind,
                "model_id": model_id,
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    for spec in MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            result = results[spec.model_id][seed]
            add(
                "training-result",
                result_paths[spec.model_id][seed],
                model_id=spec.model_id,
                seed=seed,
            )
            add(
                "training-config",
                config_paths[spec.model_id][seed],
                model_id=spec.model_id,
                seed=seed,
            )
            add(
                "checkpoint",
                referenced_path(str(result["checkpoint"])),
                model_id=spec.model_id,
                seed=seed,
            )
    for seed in CONTROLLED_SEEDS:
        add("pruning-selection", selection_paths[seed], seed=seed)
    add("compile-result", compile_result_path, seed=2026)
    add("compile-summary", compile_summary_path, seed=2026)
    add("compile-checkpoint-records", compile_records_path, seed=2026)
    for path_key, sha_key, kind in (
        ("rows", "rows_sha256", "compile-rows"),
        ("compile_indices", "compile_indices_file_sha256", "compile-indices"),
    ):
        if path_key not in compile_result:
            continue
        path = referenced_path(str(compile_result[path_key]))
        if not path.is_file():
            raise ValueError(f"missing referenced {kind}: {path}")
        if sha_key in compile_result and sha256_file(path) != str(
            compile_result[sha_key]
        ):
            raise ValueError(f"{kind} hash mismatch: {path}")
        add(kind, path, seed=2026)
    return rows


def summarize_m3() -> Path:
    (
        compile_result_path,
        compile_result,
        compile_summary_path,
        raw_compile_summary,
        compile_records_path,
        checkpoint_records,
    ) = load_compile_inputs()
    results, result_paths, configs, config_paths = load_training_inputs(
        checkpoint_records
    )
    selections, selection_paths, selection_rows = load_selection_inputs(
        results, result_paths
    )
    training_rows, model_summary = training_rows_and_summary(
        results, result_paths, configs, selections
    )
    paired_rows, paired_summary = paired_rows_and_summary(results)
    compile_rows, depth_rows, compile_summary = compile_rows_and_summary(
        compile_result, raw_compile_summary
    )
    source_rows = source_artifact_rows(
        results=results,
        result_paths=result_paths,
        config_paths=config_paths,
        selection_paths=selection_paths,
        compile_result_path=compile_result_path,
        compile_result=compile_result,
        compile_summary_path=compile_summary_path,
        compile_records_path=compile_records_path,
    )

    dataset_hashes = sorted(
        {
            str(results[spec.model_id][seed]["dataset_hash"])
            for spec in MODEL_SPECS
            for seed in CONTROLLED_SEEDS
        }
    )
    pruned = MODEL_BY_ID["l4-to-l3-pruned"]
    base = MODEL_BY_ID["l4-base"]
    layer_reduction = (base.layers - pruned.layers) / base.layers
    parameter_reduction = (
        base.parameter_count - pruned.parameter_count
    ) / base.parameter_count
    paired_accuracy_mean = float(
        paired_summary["pruned_minus_base"]["test_accuracy"]["mean"]
    )
    selection_contract_passed = all(
        bool(row["selection_rule_verified"]) and bool(row["base_loss_matches"])
        for row in selection_rows
    )
    initialization_contract_passed = all(
        bool(row["initialization_contract_verified"]) for row in training_rows
    )
    scientific_conditions = {
        "layers_reduced_exactly_25_percent": math.isclose(
            layer_reduction, 0.25, rel_tol=0.0, abs_tol=1e-12
        ),
        "parameters_reduced_exactly_25_percent": math.isclose(
            parameter_reduction, 0.25, rel_tol=0.0, abs_tol=1e-12
        ),
        "mean_test_accuracy_drop_not_over_0_005": (
            paired_accuracy_mean >= -ACCURACY_DROP_TOLERANCE
        ),
        "level0_all_seed_median_depth_nonincrease": compile_summary[
            "level0_pruned_vs_base"
        ]["all_five_nonincrease"],
        "level0_at_least_four_seed_median_depth_strict_decrease": (
            compile_summary["level0_pruned_vs_base"]["strict_decrease_count"]
            >= 4
        ),
        "compile_audit_complete_at_100_points_per_checkpoint": (
            compile_summary["completeness"]["condition_passed"]
        ),
        "probability_parity_below_1e_10": compile_summary["parity"][
            "probability_condition_passed"
        ],
        "exact_label_parity": compile_summary["parity"][
            "label_condition_passed"
        ],
    }
    pruning_effective = all(scientific_conditions.values())
    artifact_contract_conditions = {
        "one_dataset_hash": len(dataset_hashes) == 1,
        "selection_rule_and_base_loss_verified": selection_contract_passed,
        "checkpoint_lineage_and_initialization_verified": (
            initialization_contract_passed
        ),
        "compile_result_not_marked_failed": (
            compile_result["verification"] == "artifacts-verified"
        ),
    }
    failed_conditions = [
        key for key, passed in scientific_conditions.items() if not passed
    ]
    failed_artifact_contract_conditions = [
        key
        for key, passed in artifact_contract_conditions.items()
        if not passed
    ]

    output = unique_summary_dir()
    generated_csv = {
        "training_runs.csv": training_rows,
        "pruning_selection.csv": selection_rows,
        "paired_comparisons.csv": paired_rows,
        "compile_resources.csv": compile_rows,
        "compile_depth_comparisons.csv": depth_rows,
        "source_artifacts.csv": source_rows,
    }
    for filename, rows in generated_csv.items():
        write_csv(output / filename, rows)
    generated_csv_hashes = {
        filename: sha256_file(output / filename)
        for filename in generated_csv
    }

    summary = {
        "schema_version": 1,
        "stage": "M3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "dataset_hashes": dataset_hashes,
        "models": model_summary,
        "selected_layers": {
            str(seed): {
                "zero_based": int(
                    selections[seed]["selected_layer_zero_based"]
                ),
                "one_based": int(
                    selections[seed]["selected_layer_one_based"]
                ),
                "selection_result": str(selection_paths[seed]),
            }
            for seed in CONTROLLED_SEEDS
        },
        "paired_differences": paired_summary,
        "resource_reduction": {
            "base_layers": base.layers,
            "pruned_layers": pruned.layers,
            "layer_reduction_fraction": layer_reduction,
            "base_parameter_count": base.parameter_count,
            "pruned_parameter_count": pruned.parameter_count,
            "parameter_reduction_fraction": parameter_reduction,
        },
        "compile_audit": {
            **compile_summary,
            "result_path": str(compile_result_path),
            "result_sha256": sha256_file(compile_result_path),
            "summary_path": str(compile_summary_path),
            "summary_sha256": sha256_file(compile_summary_path),
        },
        "success_evaluation": {
            "accuracy_drop_tolerance": ACCURACY_DROP_TOLERANCE,
            "conditions": scientific_conditions,
            "pruning_effective": pruning_effective,
            "status": "passed" if pruning_effective else "failed",
            "failed_conditions": failed_conditions,
            "pruned_better_than_truncate_threshold": (
                ACCURACY_DROP_TOLERANCE
            ),
            "pruned_better_than_truncate": (
                paired_summary["pruned_minus_truncate"]["test_accuracy"][
                    "mean"
                ]
                > ACCURACY_DROP_TOLERANCE
            ),
            "level3_native_depth_reduction_observed": (
                compile_summary["level3_native_depth_changes"][
                    "pruned_minus_base"
                ]["any_strict_decrease"]
            ),
            "level3_all_seed_native_depth_nonincrease": (
                compile_summary["level3_native_depth_changes"][
                    "pruned_minus_base"
                ]["all_nonincrease"]
            ),
        },
        "artifact_contract": {
            "conditions": artifact_contract_conditions,
            "all_passed": all(artifact_contract_conditions.values()),
            "failed_conditions": failed_artifact_contract_conditions,
        },
        "optimizer_success": {
            model_id: {
                "success_count": details["optimizer_success_count"],
                "total": len(CONTROLLED_SEEDS),
                "all_successful": (
                    details["optimizer_success_count"]
                    == len(CONTROLLED_SEEDS)
                ),
            }
            for model_id, details in model_summary.items()
        },
        "code_revision": git_revision(),
        "generated_csv_sha256": generated_csv_hashes,
        "sha256_manifest": "SHA256SUMS",
    }
    summary_path = output / "summary.json"
    json_dump(summary_path, summary)

    manifest_entries = {
        **generated_csv_hashes,
        "summary.json": sha256_file(summary_path),
    }
    manifest_path = output / "SHA256SUMS"
    with manifest_path.open("x", encoding="utf-8") as handle:
        for filename in sorted(manifest_entries):
            handle.write(f"{manifest_entries[filename]}  {filename}\n")

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "sha256_manifest": str(manifest_path),
                "pruning_effective": pruning_effective,
                "failed_conditions": failed_conditions,
                "artifact_contract_passed": all(
                    artifact_contract_conditions.values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary_path


def main() -> None:
    summarize_m3()


if __name__ == "__main__":
    main()
