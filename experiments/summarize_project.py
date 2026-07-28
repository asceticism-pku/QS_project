#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from qs_project.core import CONTROLLED_SEEDS, load_checkpoint, sha256_file
from qs_project.training import find_verified_results, load_result


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def unique_summary_dir(stage: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = ROOT / "results" / "summary" / stage / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latest_single(config_id: str, seed: int = 30) -> dict[str, Any]:
    found = find_verified_results(config_id=config_id, seeds=(seed,))
    if seed not in found:
        raise ValueError(f"missing verified result: {config_id}, seed={seed}")
    return load_result(found[seed])


def summarize_m1() -> Path:
    output = unique_summary_dir("M1")
    loss_ids = ("legacy_amplitude", "paper_squared")
    result_paths = {
        loss_id: find_verified_results(
            config_id=f"1q-l4-{loss_id}",
            seeds=CONTROLLED_SEEDS,
            loss_id=loss_id,
        )
        for loss_id in loss_ids
    }
    for loss_id, paths in result_paths.items():
        missing = sorted(set(CONTROLLED_SEEDS) - set(paths))
        if missing:
            raise ValueError(f"{loss_id} missing seeds: {missing}")

    run_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    dataset_hashes: set[str] = set()
    training_source_hashes: set[str] = set()
    initial_parity: dict[int, bool] = {}
    results: dict[str, dict[int, dict[str, Any]]] = {
        loss_id: {} for loss_id in loss_ids
    }
    for loss_id in loss_ids:
        for seed in CONTROLLED_SEEDS:
            payload = load_result(result_paths[loss_id][seed])
            results[loss_id][seed] = payload
            dataset_hashes.add(payload["dataset_hash"])
            training_source_hashes.add(
                payload["code_revision"]["training_source_sha256"]
            )
            run_rows.append(
                {
                    "seed": seed,
                    "loss_id": loss_id,
                    "train_accuracy": payload["train_metrics"]["accuracy"],
                    "test_accuracy": payload["test_metrics"]["accuracy"],
                    "test_mean_true_margin": payload["test_metrics"][
                        "mean_true_margin"
                    ],
                    "final_loss": payload["optimizer"]["fun"],
                    "success": payload["optimizer"]["success"],
                    "status": payload["optimizer"]["status"],
                    "nfev": payload["optimizer"]["nfev"],
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "raw_result_path": payload["raw_result_path"],
                }
            )

    for seed in CONTROLLED_SEEDS:
        amplitude = results["legacy_amplitude"][seed]
        squared = results["paper_squared"][seed]
        amp_initial = load_checkpoint(Path(amplitude["initial_checkpoint"]))
        sq_initial = load_checkpoint(Path(squared["initial_checkpoint"]))
        same_initial = np.array_equal(amp_initial[0], sq_initial[0]) and np.array_equal(
            amp_initial[1], sq_initial[1]
        )
        initial_parity[seed] = bool(same_initial)
        paired_rows.append(
            {
                "seed": seed,
                "amplitude_test_accuracy": amplitude["test_metrics"]["accuracy"],
                "squared_test_accuracy": squared["test_metrics"]["accuracy"],
                "delta_test_accuracy_squared_minus_amplitude": squared[
                    "test_metrics"
                ]["accuracy"]
                - amplitude["test_metrics"]["accuracy"],
                "amplitude_nfev": amplitude["optimizer"]["nfev"],
                "squared_nfev": squared["optimizer"]["nfev"],
                "delta_nfev_squared_minus_amplitude": squared["optimizer"]["nfev"]
                - amplitude["optimizer"]["nfev"],
                "same_initial_parameters": same_initial,
            }
        )

    write_csv(output / "controlled_runs.csv", run_rows)
    write_csv(output / "paired_loss_comparison.csv", paired_rows)
    trend = [
        latest_single(f"author-weighted-1q-l{layers}") for layers in (1, 2, 4, 8)
    ]
    reference = latest_single("author-amplitude-1q-l4")
    delta_accuracy = [
        float(row["delta_test_accuracy_squared_minus_amplitude"])
        for row in paired_rows
    ]
    summary = {
        "schema_version": 1,
        "stage": "M1",
        "dataset_hashes": sorted(dataset_hashes),
        "training_source_hashes": sorted(training_source_hashes),
        "all_controlled_dataset_hashes_equal": len(dataset_hashes) == 1,
        "all_controlled_source_hashes_equal": len(training_source_hashes) == 1,
        "paired_initial_parameter_parity": initial_parity,
        "all_paired_initial_parameters_equal": all(initial_parity.values()),
        "controlled": {
            loss_id: {
                "test_accuracy": mean_sd(
                    [
                        results[loss_id][seed]["test_metrics"]["accuracy"]
                        for seed in CONTROLLED_SEEDS
                    ]
                ),
                "nfev": mean_sd(
                    [
                        float(results[loss_id][seed]["optimizer"]["nfev"])
                        for seed in CONTROLLED_SEEDS
                    ]
                ),
                "converged_count": sum(
                    bool(results[loss_id][seed]["optimizer"]["success"])
                    for seed in CONTROLLED_SEEDS
                ),
            }
            for loss_id in loss_ids
        },
        "paired_test_accuracy_delta_squared_minus_amplitude": mean_sd(
            delta_accuracy
        ),
        "practical_effect_threshold": 0.005,
        "practical_effect": abs(statistics.mean(delta_accuracy)) >= 0.005,
        "author_layer_trend": [
            {
                "config_id": row["config_id"],
                "test_accuracy": row["test_metrics"]["accuracy"],
                "final_loss": row["optimizer"]["fun"],
                "success": row["optimizer"]["success"],
                "status": row["optimizer"]["status"],
                "nfev": row["optimizer"]["nfev"],
                "raw_result_path": row["raw_result_path"],
            }
            for row in trend
        ],
        "author_amplitude_reference": {
            "test_accuracy": reference["test_metrics"]["accuracy"],
            "final_loss": reference["optimizer"]["fun"],
            "success": reference["optimizer"]["success"],
            "status": reference["optimizer"]["status"],
            "nfev": reference["optimizer"]["nfev"],
            "raw_result_path": reference["raw_result_path"],
        },
        "source_files": {
            "controlled_runs.csv": sha256_file(output / "controlled_runs.csv"),
            "paired_loss_comparison.csv": sha256_file(
                output / "paired_loss_comparison.csv"
            ),
        },
    }
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary": str(output / "summary.json"), **summary}, indent=2))
    return output / "summary.json"


def summarize_m2() -> Path:
    output = unique_summary_dir("M2")
    config_ids = (
        "1q-l4-paper_squared",
        "1q-l2-paper_squared",
        "2q-l2-separable-paper_squared",
        "2q-l2-cz-paper_squared",
    )
    architecture = {
        "1q-l4-paper_squared": {
            "n_qubits": 1,
            "layers": 4,
            "parameters": 20,
            "template_cz": 0,
        },
        "1q-l2-paper_squared": {
            "n_qubits": 1,
            "layers": 2,
            "parameters": 10,
            "template_cz": 0,
        },
        "2q-l2-separable-paper_squared": {
            "n_qubits": 2,
            "layers": 2,
            "parameters": 20,
            "template_cz": 0,
        },
        "2q-l2-cz-paper_squared": {
            "n_qubits": 2,
            "layers": 2,
            "parameters": 20,
            "template_cz": 1,
        },
    }
    paths = {
        config_id: find_verified_results(
            config_id=config_id,
            seeds=CONTROLLED_SEEDS,
            loss_id="paper_squared",
        )
        for config_id in config_ids
    }
    for config_id, found in paths.items():
        missing = sorted(set(CONTROLLED_SEEDS) - set(found))
        if missing:
            raise ValueError(f"{config_id} missing seeds: {missing}")
    results = {
        config_id: {
            seed: load_result(paths[config_id][seed]) for seed in CONTROLLED_SEEDS
        }
        for config_id in config_ids
    }
    run_rows: list[dict[str, Any]] = []
    for config_id in config_ids:
        for seed in CONTROLLED_SEEDS:
            row = results[config_id][seed]
            run_rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    **architecture[config_id],
                    "train_accuracy": row["train_metrics"]["accuracy"],
                    "test_accuracy": row["test_metrics"]["accuracy"],
                    "test_mean_true_margin": row["test_metrics"]["mean_true_margin"],
                    "final_loss": row["optimizer"]["fun"],
                    "success": row["optimizer"]["success"],
                    "status": row["optimizer"]["status"],
                    "nfev": row["optimizer"]["nfev"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "raw_result_path": row["raw_result_path"],
                }
            )
    write_csv(output / "training_runs.csv", run_rows)

    paired_rows: list[dict[str, Any]] = []
    paired_initial_parity: dict[int, dict[str, bool]] = {}
    for seed in CONTROLLED_SEEDS:
        base = results["1q-l4-paper_squared"][seed]
        sep = results["2q-l2-separable-paper_squared"][seed]
        cz = results["2q-l2-cz-paper_squared"][seed]
        base_initial = load_checkpoint(Path(base["initial_checkpoint"]))
        sep_initial = load_checkpoint(Path(sep["initial_checkpoint"]))
        cz_initial = load_checkpoint(Path(cz["initial_checkpoint"]))
        parity = {
            "l4_vs_2q_separable_flat": bool(
                np.array_equal(base_initial[0].ravel(), sep_initial[0].ravel())
                and np.array_equal(base_initial[1].ravel(), sep_initial[1].ravel())
            ),
            "2q_separable_vs_cz": bool(
                np.array_equal(sep_initial[0], cz_initial[0])
                and np.array_equal(sep_initial[1], cz_initial[1])
            ),
        }
        paired_initial_parity[seed] = parity
        paired_rows.append(
            {
                "seed": seed,
                "l4_test_accuracy": base["test_metrics"]["accuracy"],
                "l2_test_accuracy": results["1q-l2-paper_squared"][seed][
                    "test_metrics"
                ]["accuracy"],
                "separable_test_accuracy": sep["test_metrics"]["accuracy"],
                "cz_test_accuracy": cz["test_metrics"]["accuracy"],
                "l2_minus_l4": results["1q-l2-paper_squared"][seed]["test_metrics"][
                    "accuracy"
                ]
                - base["test_metrics"]["accuracy"],
                "separable_minus_l4": sep["test_metrics"]["accuracy"]
                - base["test_metrics"]["accuracy"],
                "cz_minus_l4": cz["test_metrics"]["accuracy"]
                - base["test_metrics"]["accuracy"],
                "cz_minus_separable": cz["test_metrics"]["accuracy"]
                - sep["test_metrics"]["accuracy"],
                **parity,
            }
        )
    write_csv(output / "paired_architecture_comparison.csv", paired_rows)

    compile_results = sorted(
        (ROOT / "results" / "raw" / "M2" / "compile-audit").glob(
            "seed-2026/*/result.json"
        )
    )
    if not compile_results:
        raise ValueError("missing M2 compile audit")
    compile_result_path = compile_results[-1]
    compile_result = json.loads(compile_result_path.read_text(encoding="utf-8"))
    compile_summary = json.loads(
        Path(compile_result["summary"]).read_text(encoding="utf-8")
    )
    resource_rows: list[dict[str, Any]] = []
    for config_id in config_ids:
        for level in (0, 3):
            selected = [
                row
                for row in compile_summary
                if row["config_id"] == config_id
                and row["optimization_level"] == level
            ]
            resource_rows.append(
                {
                    "config_id": config_id,
                    "optimization_level": level,
                    **architecture[config_id],
                    "median_target_depth_across_seed_medians": float(
                        statistics.median(
                            row["median_target_depth"] for row in selected
                        )
                    ),
                    "median_rz_across_seed_medians": float(
                        statistics.median(row["median_rz"] for row in selected)
                    ),
                    "median_sx_across_seed_medians": float(
                        statistics.median(row["median_sx"] for row in selected)
                    ),
                    "median_x_across_seed_medians": float(
                        statistics.median(row["median_x"] for row in selected)
                    ),
                    "median_cz_across_seed_medians": float(
                        statistics.median(row["median_cz"] for row in selected)
                    ),
                    "max_probability_error": max(
                        row["max_probability_error"] for row in selected
                    ),
                    "all_probability_parity": all(
                        row["all_probability_parity"] for row in selected
                    ),
                    "all_label_parity": all(
                        row["all_label_parity"] for row in selected
                    ),
                }
            )
    write_csv(output / "resource_summary.csv", resource_rows)

    model_summary = {
        config_id: {
            "test_accuracy": mean_sd(
                [
                    results[config_id][seed]["test_metrics"]["accuracy"]
                    for seed in CONTROLLED_SEEDS
                ]
            ),
            "nfev": mean_sd(
                [
                    float(results[config_id][seed]["optimizer"]["nfev"])
                    for seed in CONTROLLED_SEEDS
                ]
            ),
            "converged_count": sum(
                bool(results[config_id][seed]["optimizer"]["success"])
                for seed in CONTROLLED_SEEDS
            ),
            **architecture[config_id],
        }
        for config_id in config_ids
    }
    paired_deltas = {
        name: mean_sd([float(row[name]) for row in paired_rows])
        for name in (
            "l2_minus_l4",
            "separable_minus_l4",
            "cz_minus_l4",
            "cz_minus_separable",
        )
    }
    summary = {
        "schema_version": 1,
        "stage": "M2",
        "models": model_summary,
        "paired_accuracy_deltas": paired_deltas,
        "performance_similar_threshold": 0.005,
        "performance_similar": {
            name: abs(values["mean"]) <= 0.005
            for name, values in paired_deltas.items()
        },
        "paired_initial_parameter_parity": paired_initial_parity,
        "all_parameter_matched_initializations_equal": all(
            all(values.values()) for values in paired_initial_parity.values()
        ),
        "compile_audit": {
            **compile_result,
            "result_path": str(compile_result_path),
        },
        "source_files": {
            "training_runs.csv": sha256_file(output / "training_runs.csv"),
            "paired_architecture_comparison.csv": sha256_file(
                output / "paired_architecture_comparison.csv"
            ),
            "resource_summary.csv": sha256_file(output / "resource_summary.csv"),
        },
    }
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary": str(output / "summary.json"), **summary}, indent=2))
    return output / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("m1", "m2"))
    args = parser.parse_args()
    if args.stage == "m1":
        summarize_m1()
    elif args.stage == "m2":
        summarize_m2()


if __name__ == "__main__":
    main()
