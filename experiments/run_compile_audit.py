#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from qs_project.compile_audit import audit_samples
from qs_project.core import (
    CONTROLLED_SEEDS,
    canonical_json_hash,
    create_run_directory,
    git_revision,
    json_dump,
    load_checkpoint,
    make_circle_dataset,
    sha256_file,
)
from qs_project.shots import stratified_eval_indices
from qs_project.training import find_verified_results, load_result

COMPILE_SEED = 2026
SEED_TRANSPILER = 30


@dataclass(frozen=True)
class CheckpointSpec:
    model_id: str
    source_config_id: str
    qubits: int
    layers: int
    entanglement: str
    source_stage: str


def exact_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def compile_indices_artifact(
    *,
    require_existing: bool = False,
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    dataset = make_circle_dataset()
    relative_indices = stratified_eval_indices(
        dataset.test_y, per_class=50, seed=COMPILE_SEED
    )
    payload = {
        "schema_version": 1,
        "purpose": "compile-audit",
        "compile_seed": COMPILE_SEED,
        "sampling": "stratified-without-replacement",
        "per_class": 50,
        "index_space": "test-set-relative",
        "test_relative_indices": relative_indices.tolist(),
        "dataset_global_indices": (relative_indices + 200).tolist(),
        "labels": dataset.test_y[relative_indices].astype(int).tolist(),
        "dataset_hash": dataset.dataset_hash,
    }
    payload["indices_content_sha256"] = canonical_json_hash(payload)
    path = ROOT / "results" / "indices" / "compile_indices.json"
    if require_existing and not path.is_file():
        raise FileNotFoundError(
            f"M3 must reuse the existing compile index artifact: {path}"
        )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing compile indices conflict with contract: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        json_dump(path, payload)
    return path, relative_indices, payload


def label_from_probabilities(probabilities: dict[str, float], n_qubits: int) -> int:
    zero = "0" * n_qubits
    one = "1" * n_qubits
    return 0 if probabilities.get(zero, 0.0) >= probabilities.get(one, 0.0) else 1


def checkpoint_specs(stage: str) -> tuple[CheckpointSpec, ...]:
    if stage == "m2":
        return (
            CheckpointSpec(
                "1q-l4-paper_squared",
                "1q-l4-paper_squared",
                1,
                4,
                "n",
                "M1",
            ),
            CheckpointSpec(
                "1q-l2-paper_squared",
                "1q-l2-paper_squared",
                1,
                2,
                "n",
                "M2",
            ),
            CheckpointSpec(
                "2q-l2-separable-paper_squared",
                "2q-l2-separable-paper_squared",
                2,
                2,
                "n",
                "M2",
            ),
            CheckpointSpec(
                "2q-l2-cz-paper_squared",
                "2q-l2-cz-paper_squared",
                2,
                2,
                "y",
                "M2",
            ),
        )
    if stage == "m3":
        return (
            CheckpointSpec(
                "l4-base",
                "1q-l4-paper_squared",
                1,
                4,
                "n",
                "M1",
            ),
            CheckpointSpec(
                "l4-to-l3-pruned",
                "l4-to-l3-pruned",
                1,
                3,
                "n",
                "M3",
            ),
            CheckpointSpec(
                "l4-truncate-last",
                "l4-truncate-last",
                1,
                3,
                "n",
                "M3",
            ),
            CheckpointSpec(
                "l3-scratch",
                "l3-scratch",
                1,
                3,
                "n",
                "M3",
            ),
        )
    raise ValueError(f"unsupported compile stage: {stage}")


def _summary_row(
    *,
    spec: CheckpointSpec,
    seed: int,
    optimization_level: int,
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    level_rows = [
        row
        for row in audit_rows
        if row["optimization_level"] == optimization_level
    ]
    if not level_rows:
        raise ValueError(
            f"{spec.model_id} seed {seed} has no level "
            f"{optimization_level} compile rows"
        )
    return {
        "config_id": spec.model_id,
        "model_id": spec.model_id,
        "source_config_id": spec.source_config_id,
        "source_stage": spec.source_stage,
        "seed": seed,
        "optimization_level": optimization_level,
        "n_points": len(level_rows),
        "median_target_depth": float(
            statistics.median(row["depth_no_measure"] for row in level_rows)
        ),
        "median_rz": float(
            statistics.median(row["rz_count"] for row in level_rows)
        ),
        "median_sx": float(
            statistics.median(row["sx_count"] for row in level_rows)
        ),
        "median_x": float(
            statistics.median(row["x_count"] for row in level_rows)
        ),
        "median_cz": float(
            statistics.median(row["cz_count"] for row in level_rows)
        ),
        "max_probability_error": max(
            row["max_probability_error"] for row in level_rows
        ),
        "all_probability_parity": all(
            row["parity_passed"] for row in level_rows
        ),
        "all_label_parity": all(
            row["exact_label_match"] for row in level_rows
        ),
    }


def _write_rows(rows_path: Path, rows: list[dict[str, Any]]) -> None:
    with rows_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _revision_record() -> dict[str, Any]:
    revision = git_revision()
    revision["compile_runner_sha256"] = sha256_file(Path(__file__).resolve())
    revision["compile_module_sha256"] = sha256_file(
        ROOT / "src" / "qs_project" / "compile_audit.py"
    )
    return revision


def _finalize_artifacts(
    *,
    stage: str,
    dataset_hash: str | None,
    indices_path: Path,
    indices_payload: dict[str, Any] | None,
    point_count: int,
    run_dir: Path,
    rows_path: Path,
    checkpoint_records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    row_count: int,
    max_error: float | None,
    all_probability_parity: bool,
    all_label_parity: bool,
    failure: BaseException | None,
) -> Path:
    checkpoint_records_path = run_dir / "checkpoint_records.json"
    summary_path = run_dir / "compile_summary.json"
    json_dump(checkpoint_records_path, checkpoint_records)
    json_dump(summary_path, summary_rows)

    if failure is not None:
        verification = "failed-exception"
    elif all_probability_parity and all_label_parity:
        verification = "artifacts-verified"
    else:
        verification = "parity-failed"

    specs = checkpoint_specs(stage)
    result_payload: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": stage.upper(),
        "config_id": "compile-audit",
        "compile_seed": COMPILE_SEED,
        "seed_transpiler": SEED_TRANSPILER,
        "dataset_hash": dataset_hash,
        "compile_indices": str(indices_path),
        "compile_indices_file_sha256": (
            sha256_file(indices_path) if indices_path.is_file() else None
        ),
        "compile_indices_content_sha256": (
            indices_payload["indices_content_sha256"]
            if indices_payload is not None
            else None
        ),
        "models": [spec.model_id for spec in specs],
        "source_config_ids": [spec.source_config_id for spec in specs],
        "expected_checkpoint_count": len(specs) * len(CONTROLLED_SEEDS),
        "checkpoint_count": len(checkpoint_records),
        "point_count_per_checkpoint": point_count,
        "row_count": row_count,
        "max_probability_error": max_error,
        "all_probability_parity": all_probability_parity,
        "all_label_parity": all_label_parity,
        "verification": verification,
        "evidence_label": "compiled-estimate",
        "command": exact_command(),
        "code_revision": _revision_record(),
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_records": str(checkpoint_records_path),
        "checkpoint_records_sha256": sha256_file(checkpoint_records_path),
        "raw_result_path": str(run_dir),
    }
    if failure is not None:
        result_payload["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
    result_path = run_dir / "result.json"
    json_dump(result_path, result_payload)
    print("RESULT " + json.dumps(result_payload, sort_keys=True), flush=True)
    return result_path


def run_compile_audit(stage: str) -> Path:
    specs = checkpoint_specs(stage)
    run_dir = create_run_directory(
        stage.upper(),
        "compile-audit",
        COMPILE_SEED,
    )
    rows_path = run_dir / "compile_rows.jsonl"
    rows_path.touch(exist_ok=False)
    checkpoint_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    indices_path = ROOT / "results" / "indices" / "compile_indices.json"
    indices_payload: dict[str, Any] | None = None
    dataset_hash: str | None = None
    point_count = 0
    row_count = 0
    max_error: float | None = None
    all_probability_parity = True
    all_label_parity = True

    try:
        dataset = make_circle_dataset()
        dataset_hash = dataset.dataset_hash
        indices_path, indices, indices_payload = compile_indices_artifact(
            require_existing=stage == "m3"
        )
        point_count = len(indices)

        resolved: list[
            tuple[
                CheckpointSpec,
                int,
                Path,
                dict[str, Any],
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for spec in specs:
            found = find_verified_results(
                config_id=spec.source_config_id,
                seeds=CONTROLLED_SEEDS,
                loss_id="paper_squared",
            )
            missing = sorted(set(CONTROLLED_SEEDS) - set(found))
            if missing:
                raise ValueError(
                    f"{spec.source_config_id} missing verified seeds {missing}"
                )
            for seed in CONTROLLED_SEEDS:
                result_path = found[seed]
                result = load_result(result_path)
                if result.get("experiment_id") != spec.source_stage:
                    raise ValueError(
                        f"{spec.source_config_id} seed {seed} came from "
                        f"{result.get('experiment_id')}, expected {spec.source_stage}"
                    )
                if result.get("dataset_hash") != dataset.dataset_hash:
                    raise ValueError(
                        f"{spec.source_config_id} seed {seed} dataset hash mismatch"
                    )
                checkpoint_path = Path(result["checkpoint"])
                expected_hash = str(result["checkpoint_sha256"])
                if sha256_file(checkpoint_path) != expected_hash:
                    raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
                theta, alpha, weights = load_checkpoint(checkpoint_path)
                if weights is not None:
                    raise ValueError(
                        f"{spec.source_config_id} seed {seed} is unexpectedly weighted"
                    )
                if theta.shape != (spec.qubits, spec.layers, 3):
                    raise ValueError(
                        f"{spec.source_config_id} seed {seed} theta shape "
                        f"{theta.shape} != {(spec.qubits, spec.layers, 3)}"
                    )
                if alpha.shape != (spec.qubits, spec.layers, 2):
                    raise ValueError(
                        f"{spec.source_config_id} seed {seed} alpha shape "
                        f"{alpha.shape} != {(spec.qubits, spec.layers, 2)}"
                    )
                checkpoint_records.append(
                    {
                        "config_id": spec.model_id,
                        "model_id": spec.model_id,
                        "source_config_id": spec.source_config_id,
                        "source_stage": spec.source_stage,
                        "seed": seed,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": expected_hash,
                        "source_result": str(result_path),
                    }
                )
                resolved.append(
                    (spec, seed, checkpoint_path, result, theta, alpha)
                )

        for spec, seed, checkpoint_path, result, theta, alpha in resolved:
            audit_rows = audit_samples(
                theta,
                alpha,
                dataset.test_x[indices],
                entanglement=spec.entanglement,
                sample_ids=indices.tolist(),
                optimization_levels=(0, 3),
                seed_transpiler=SEED_TRANSPILER,
                parity_tolerance=1e-10,
                include_measurements=False,
            )
            for row in audit_rows:
                row.update(
                    {
                        "config_id": spec.model_id,
                        "model_id": spec.model_id,
                        "source_config_id": spec.source_config_id,
                        "training_seed": seed,
                        "source_stage": spec.source_stage,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": result["checkpoint_sha256"],
                    }
                )
                row["logical_label"] = label_from_probabilities(
                    row["logical_probabilities"], spec.qubits
                )
                row["compiled_label"] = label_from_probabilities(
                    row["compiled_probabilities"], spec.qubits
                )
                row["exact_label_match"] = (
                    row["logical_label"] == row["compiled_label"]
                )
            _write_rows(rows_path, audit_rows)
            row_count += len(audit_rows)
            checkpoint_max_error = max(
                row["max_probability_error"] for row in audit_rows
            )
            max_error = (
                checkpoint_max_error
                if max_error is None
                else max(max_error, checkpoint_max_error)
            )
            all_probability_parity = all_probability_parity and all(
                row["parity_passed"] for row in audit_rows
            )
            all_label_parity = all_label_parity and all(
                row["exact_label_match"] for row in audit_rows
            )
            for level in (0, 3):
                summary_rows.append(
                    _summary_row(
                        spec=spec,
                        seed=seed,
                        optimization_level=level,
                        audit_rows=audit_rows,
                    )
                )
            print(
                "COMPILE_PROGRESS "
                + json.dumps(
                    {
                        "config_id": spec.model_id,
                        "source_config_id": spec.source_config_id,
                        "seed": seed,
                        "rows": len(audit_rows),
                        "max_probability_error": checkpoint_max_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        expected_rows = len(resolved) * point_count * 2
        if row_count != expected_rows:
            raise ValueError(
                f"compile row count {row_count} != expected {expected_rows}"
            )
        if len(summary_rows) != len(resolved) * 2:
            raise ValueError("compile summary row count is incomplete")
    except BaseException as failure:
        _finalize_artifacts(
            stage=stage,
            dataset_hash=dataset_hash,
            indices_path=indices_path,
            indices_payload=indices_payload,
            point_count=point_count,
            run_dir=run_dir,
            rows_path=rows_path,
            checkpoint_records=checkpoint_records,
            summary_rows=summary_rows,
            row_count=row_count,
            max_error=max_error,
            all_probability_parity=False,
            all_label_parity=False,
            failure=failure,
        )
        raise

    result_path = _finalize_artifacts(
        stage=stage,
        dataset_hash=dataset_hash,
        indices_path=indices_path,
        indices_payload=indices_payload,
        point_count=point_count,
        run_dir=run_dir,
        rows_path=rows_path,
        checkpoint_records=checkpoint_records,
        summary_rows=summary_rows,
        row_count=row_count,
        max_error=max_error,
        all_probability_parity=all_probability_parity,
        all_label_parity=all_label_parity,
        failure=None,
    )
    if not all_probability_parity or not all_label_parity:
        raise SystemExit("compile parity failed; raw artifacts were preserved")
    return result_path


def run_m2_compile_audit() -> Path:
    return run_compile_audit("m2")


def run_m3_compile_audit() -> Path:
    return run_compile_audit("m3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("m2", "m3"))
    args = parser.parse_args()
    if args.stage == "m2":
        run_m2_compile_audit()
    elif args.stage == "m3":
        run_m3_compile_audit()


if __name__ == "__main__":
    main()
