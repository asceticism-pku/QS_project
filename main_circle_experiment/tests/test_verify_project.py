from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments import verify_project as verifier
from qs_project.core import sha256_file


DATASET_HASH = "d" * 64
REVISION = {
    "head": "0d647b0e8019d3b0ea59baf1af82b51a08bb6448",
    "branch": "main",
    "dirty": True,
    "tracked_diff_sha256": "a" * 64,
    "training_source_sha256": "b" * 64,
}
COMMAND = "/fixture/.venv/bin/python experiments/run_project.py"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_optimizer_run(
    *,
    raw_root: Path,
    spec: verifier.ExpectedRun,
    dataset_path: Path,
    leaf: str = "20260728T000000.000000Z-fixture",
) -> Path:
    run_dir = (
        raw_root
        / spec.experiment_id
        / spec.config_id
        / f"seed-{spec.seed}"
        / leaf
    )
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.npz"
    initial_checkpoint = run_dir / "initial_checkpoint.npz"
    checkpoint.write_bytes(
        f"checkpoint:{spec.identity}:{leaf}".encode()
    )
    initial_checkpoint.write_bytes(
        f"initial:{spec.identity}:{leaf}".encode()
    )

    maxfun = 30000 if spec.config_id == "l3-scratch" else 15000
    config = {
        "experiment_id": spec.experiment_id,
        "config_id": spec.config_id,
        "qubits": spec.qubits,
        "layers": spec.layers,
        "entanglement": spec.entanglement,
        "loss_id": spec.loss_id,
        "rng_mode": spec.rng_mode,
        "init_seed": spec.seed,
        "data_seed": 30,
        "maxfun": maxfun,
        "maxiter": maxfun,
        "ftol": 2.22e-9,
        "gtol": 1e-5,
        "evidence_label": "ideal-simulation",
        "run_kind": spec.run_kind,
        "parent_checkpoint": None,
        "removed_layer": None,
        "optimizer": "L-BFGS-B",
        "optimizer_options": None,
        "dataset_hash": DATASET_HASH,
        "dataset_path": str(dataset_path),
        "train_count": 200,
        "test_count": 4000,
        "parameter_count": spec.qubits * spec.layers * 5,
        "command": COMMAND,
        "code_revision": REVISION,
    }
    config["config_fingerprint"] = verifier._config_fingerprint(config)
    config_path = run_dir / "config.json"
    write_json(config_path, config)
    (run_dir / "command.txt").write_text(COMMAND + "\n", encoding="utf-8")

    if spec.known_maxfun_nonconvergence:
        success = False
        status = 1
        nfev = maxfun + 7
        message = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
    else:
        success = True
        status = 0
        nfev = 100
        message = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    result = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "config_id": spec.config_id,
        "seed": spec.seed,
        "dataset_hash": DATASET_HASH,
        "config_fingerprint": config["config_fingerprint"],
        "loss_id": spec.loss_id,
        "rng_mode": spec.rng_mode,
        "evidence_label": "ideal-simulation",
        "verification": "artifacts-verified",
        "optimizer": {
            "success": success,
            "status": status,
            "message": message,
            "fun": 0.2,
            "initial_fun": 0.5,
            "objective_delta": -0.3,
            "nfev": nfev,
            "nit": 10,
            "njev": 10,
            "elapsed_seconds": 0.1,
        },
        "train_metrics": {
            "accuracy": 0.9,
            "mean_true_margin": 0.5,
            "mean_abs_decision_margin": 0.6,
            "min_abs_decision_margin": 0.01,
        },
        "test_metrics": {
            "accuracy": 0.88,
            "mean_true_margin": 0.4,
            "mean_abs_decision_margin": 0.5,
            "min_abs_decision_margin": 0.001,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "code_revision": REVISION,
        "command": COMMAND,
        "raw_result_path": str(run_dir),
    }
    result_path = run_dir / "result.json"
    write_json(result_path, result)
    return result_path


def write_compile_result(
    *,
    root: Path,
    raw_root: Path,
    stage: str,
    singleton_runs: dict[tuple[str, str, int], verifier.SingletonRun],
) -> Path:
    run_dir = (
        raw_root
        / stage
        / "compile-audit"
        / "seed-2026"
        / "20260728T010000.000000Z-fixture"
    )
    run_dir.mkdir(parents=True)
    rows_path = run_dir / "compile_rows.jsonl"
    summary_path = run_dir / "compile_summary.json"
    records_path = run_dir / "checkpoint_records.json"
    indices_path = root / "results" / "indices" / "compile_indices.json"
    rows_path.write_text("{}\n" * 4000, encoding="utf-8")
    write_json(summary_path, [])
    write_json(indices_path, {"compile_seed": 2026})

    records = []
    for (model_id, seed), source_identity in verifier._expected_compile_sources(
        stage
    ).items():
        source = singleton_runs[source_identity]
        records.append(
            {
                "config_id": model_id,
                "model_id": model_id,
                "source_config_id": source_identity[1],
                "source_stage": source_identity[0],
                "seed": seed,
                "checkpoint": source.result["checkpoint"],
                "checkpoint_sha256": source.result["checkpoint_sha256"],
                "source_result": str(source.result_path),
            }
        )
    write_json(records_path, records)
    if stage == "M2":
        verification = "parity-failed"
        all_probability_parity = False
        max_probability_error = 4.7e-10
    else:
        verification = "artifacts-verified"
        all_probability_parity = True
        max_probability_error = 2e-15
    result = {
        "schema_version": 2,
        "experiment_id": stage,
        "config_id": "compile-audit",
        "compile_seed": 2026,
        "seed_transpiler": 30,
        "dataset_hash": DATASET_HASH,
        "checkpoint_count": 20,
        "expected_checkpoint_count": 20,
        "point_count_per_checkpoint": 100,
        "row_count": 4000,
        "max_probability_error": max_probability_error,
        "all_probability_parity": all_probability_parity,
        "all_label_parity": True,
        "verification": verification,
        "evidence_label": "compiled-estimate",
        "command": f"python experiments/run_compile_audit.py {stage.lower()}",
        "code_revision": {"head": REVISION["head"]},
        "rows": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_records": str(records_path),
        "checkpoint_records_sha256": sha256_file(records_path),
        "compile_indices": str(indices_path),
        "compile_indices_file_sha256": sha256_file(indices_path),
        "raw_result_path": str(run_dir),
    }
    result_path = run_dir / "result.json"
    write_json(result_path, result)
    return result_path


def write_m4_result(
    *,
    raw_root: Path,
    dataset_path: Path,
    singleton_runs: dict[tuple[str, str, int], verifier.SingletonRun],
) -> Path:
    run_dir = (
        raw_root
        / "M4"
        / "fixed-adaptive-shots"
        / "seed-2026"
        / "20260728T020000.000000Z-fixture"
    )
    run_dir.mkdir(parents=True)
    artifact_paths = {
        "campaign_metrics_csv": run_dir / "campaign_metrics.csv",
        "campaign_metrics_json": run_dir / "campaign_metrics.json",
        "campaign_metrics_jsonl": run_dir / "campaign_metrics.jsonl",
        "config": run_dir / "config.json",
        "environment": run_dir / "environment.json",
        "eval_indices": run_dir / "eval_indices.json",
        "exact_probabilities": run_dir / "exact_probabilities.npz",
        "metrics": run_dir / "metrics.json",
    }
    for key, path in artifact_paths.items():
        if key == "config":
            continue
        path.write_text(f"{key}\n", encoding="utf-8")

    records = []
    sources = {
        ("l4-base", seed): ("M1", "1q-l4-paper_squared", seed)
        for seed in verifier.CONTROLLED_SEEDS
    }
    sources.update(
        {
            ("l4-to-l3-pruned", seed): ("M3", "l4-to-l3-pruned", seed)
            for seed in verifier.CONTROLLED_SEEDS
        }
    )
    for (model_id, seed), source_identity in sources.items():
        source = singleton_runs[source_identity]
        records.append(
            {
                "model_id": model_id,
                "source_experiment_id": source_identity[0],
                "source_config_id": source_identity[1],
                "training_seed": seed,
                "checkpoint": source.result["checkpoint"],
                "checkpoint_recorded_path": source.result["checkpoint"],
                "checkpoint_sha256": source.result["checkpoint_sha256"],
                "source_result": str(source.result_path),
                "selected_result": str(source.result_path),
                "source_result_sha256": sha256_file(source.result_path),
                "verified_candidate_count": 1,
                "dataset_hash": DATASET_HASH,
            }
        )
    records_path = run_dir / "checkpoint_records.json"
    write_json(records_path, records)
    command = "python experiments/run_shot_evaluation.py"
    config = {
        "experiment_id": "M4",
        "config_id": "fixed-adaptive-shots",
        "dataset_hash": DATASET_HASH,
        "optimizer_runs": 0,
        "run_kind": "frozen-checkpoint-shot-evaluation",
        "command": command,
    }
    write_json(artifact_paths["config"], config)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    result = {
        "schema_version": 1,
        "experiment_id": "M4",
        "config_id": "fixed-adaptive-shots",
        "dataset_hash": DATASET_HASH,
        "verification": "artifacts-verified",
        "evidence_label": "shot-simulation",
        "optimizer_runs": 0,
        "nfev": 0,
        "checkpoint_count": 10,
        "campaign_repeats_per_checkpoint": 100,
        "campaign_count": 1000,
        "campaign_metric_row_count": 4000,
        "evaluation_source_stable": True,
        "command": command,
        "code_revision": {"head": REVISION["head"]},
        "checkpoint_records": str(records_path),
        "checkpoint_records_sha256": sha256_file(records_path),
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "raw_result_path": str(run_dir),
    }
    for key, path in artifact_paths.items():
        result[key] = str(path)
        result[f"{key}_sha256"] = sha256_file(path)
    result["eval_indices_file_sha256"] = result.pop("eval_indices_sha256")
    result_path = run_dir / "result.json"
    write_json(result_path, result)
    return result_path


@pytest.fixture
def complete_project(tmp_path):
    raw_root = tmp_path / "results" / "raw"
    dataset_path = tmp_path / "results" / "datasets" / "circle.npz"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes(b"fixed-circle-dataset")
    for spec in verifier.expected_optimizer_runs():
        write_optimizer_run(
            raw_root=raw_root,
            spec=spec,
            dataset_path=dataset_path,
        )
    inputs = verifier.InputRegistry()
    optimizer_report, _, singleton_runs = verifier.audit_optimizer_matrix(
        raw_root=raw_root,
        dataset_hash=DATASET_HASH,
        inputs=inputs,
        repo_root=tmp_path,
    )
    assert optimizer_report["passed"] is True
    m2_result = write_compile_result(
        root=tmp_path,
        raw_root=raw_root,
        stage="M2",
        singleton_runs=singleton_runs,
    )
    m3_result = write_compile_result(
        root=tmp_path,
        raw_root=raw_root,
        stage="M3",
        singleton_runs=singleton_runs,
    )
    m4_result = write_m4_result(
        raw_root=raw_root,
        dataset_path=dataset_path,
        singleton_runs=singleton_runs,
    )
    return {
        "root": tmp_path,
        "raw_root": raw_root,
        "dataset_path": dataset_path,
        "singleton_runs": singleton_runs,
        "m2_result": m2_result,
        "m3_result": m3_result,
        "m4_result": m4_result,
    }


def test_expected_optimizer_matrix_is_exactly_15_plus_15_plus_15() -> None:
    specs = verifier.expected_optimizer_runs()

    assert len(specs) == 45
    assert {
        stage: sum(spec.experiment_id == stage for spec in specs)
        for stage in ("M1", "M2", "M3")
    } == {"M1": 15, "M2": 15, "M3": 15}
    assert sum(spec.known_maxfun_nonconvergence for spec in specs) == 1


def test_complete_fixture_writes_verified_final_audit(complete_project) -> None:
    project = complete_project
    audit_path = verifier.run_final_audit(
        raw_root=project["raw_root"],
        summary_root=project["root"] / "results" / "summary",
        expected_dataset_hash=DATASET_HASH,
        repo_root=project["root"],
        fail_on_error=True,
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["verification"] == "artifacts-verified"
    assert audit["optimizer_audit"]["actual_count"] == 45
    assert audit["optimizer_audit"]["known_l8_nonconvergence_preserved"] is True
    assert audit["M4_audit"]["optimizer_runs"] == 0
    assert audit["M4_audit"]["nfev"] == 0
    assert audit["M4_audit"]["checkpoint_reuse_verified"] is True
    m2_compile = audit["compile_audits"]["M2"]
    assert m2_compile["passed"] is True
    assert m2_compile["actual_row_count"] == 4000
    assert m2_compile["scientific_findings"] == [
        {
            "affects_artifact_integrity": False,
            "classification": "preserved-scientific-negative-result",
            "expected_for_stage": True,
            "finding": "probability-parity-failed",
            "max_probability_error": 4.7e-10,
            "observed": True,
            "tolerance": verifier.PARITY_TOLERANCE,
        }
    ]

    csv_path = audit_path.parent / "optimizer_runs.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45
    assert sum(row["known_maxfun_nonconvergence"] == "True" for row in rows) == 1
    manifest = (audit_path.parent / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"{sha256_file(audit_path)}  final_audit.json" in manifest
    assert f"{sha256_file(csv_path)}  optimizer_runs.csv" in manifest


def test_duplicate_optimizer_identity_fails_exact_matrix(complete_project) -> None:
    project = complete_project
    duplicated_spec = verifier.expected_optimizer_runs()[0]
    write_optimizer_run(
        raw_root=project["raw_root"],
        spec=duplicated_spec,
        dataset_path=project["dataset_path"],
        leaf="20260728T030000.000000Z-duplicate",
    )

    report, rows, _ = verifier.audit_optimizer_matrix(
        raw_root=project["raw_root"],
        dataset_hash=DATASET_HASH,
        inputs=verifier.InputRegistry(),
        repo_root=project["root"],
    )

    assert report["passed"] is False
    assert report["actual_count"] == 46
    assert report["exact_identity_once"] is False
    assert any("duplicate optimizer run" in issue for issue in report["issues"])
    assert len(rows) == 46


def test_m4_rejects_copied_checkpoint_even_with_same_hash(
    complete_project,
) -> None:
    project = complete_project
    result = json.loads(project["m4_result"].read_text(encoding="utf-8"))
    records_path = Path(result["checkpoint_records"])
    records = json.loads(records_path.read_text(encoding="utf-8"))
    original_checkpoint = Path(records[0]["checkpoint"])
    copied_checkpoint = project["root"] / "copied-checkpoint.npz"
    copied_checkpoint.write_bytes(original_checkpoint.read_bytes())
    records[0]["checkpoint"] = str(copied_checkpoint)
    records[0]["checkpoint_recorded_path"] = str(copied_checkpoint)
    write_json(records_path, records)
    result["checkpoint_records_sha256"] = sha256_file(records_path)
    write_json(project["m4_result"], result)

    report = verifier.audit_m4_result(
        raw_root=project["raw_root"],
        dataset_hash=DATASET_HASH,
        optimizer_runs=project["singleton_runs"],
        inputs=verifier.InputRegistry(),
        repo_root=project["root"],
    )

    assert report["passed"] is False
    assert report["checkpoint_reuse_verified"] is False
    assert any("does not reuse source checkpoint" in issue for issue in report["issues"])


def test_m2_compile_parity_failure_is_preserved_scientific_result(
    complete_project,
) -> None:
    project = complete_project

    report = verifier.audit_compile_result(
        stage="M2",
        raw_root=project["raw_root"],
        dataset_hash=DATASET_HASH,
        optimizer_runs=project["singleton_runs"],
        inputs=verifier.InputRegistry(),
        repo_root=project["root"],
    )

    assert report["passed"] is True
    assert report["verification"] == "parity-failed"
    assert report["all_probability_parity"] is False
    assert report["issues"] == []
    assert report["scientific_findings"][0]["observed"] is True
    assert (
        report["scientific_findings"][0]["classification"]
        == "preserved-scientific-negative-result"
    )


def test_m2_compile_artifact_corruption_still_fails(complete_project) -> None:
    project = complete_project
    result = json.loads(project["m2_result"].read_text(encoding="utf-8"))
    rows_path = Path(result["rows"])
    rows_path.write_text("{}\n" * 3999, encoding="utf-8")

    report = verifier.audit_compile_result(
        stage="M2",
        raw_root=project["raw_root"],
        dataset_hash=DATASET_HASH,
        optimizer_runs=project["singleton_runs"],
        inputs=verifier.InputRegistry(),
        repo_root=project["root"],
    )

    assert report["passed"] is False
    assert report["scientific_findings"][0]["observed"] is True
    assert any("SHA-256 mismatch" in issue for issue in report["issues"])
    assert any("3999 rows" in issue for issue in report["issues"])


def test_m3_compile_still_requires_probability_parity(complete_project) -> None:
    project = complete_project
    result = json.loads(project["m3_result"].read_text(encoding="utf-8"))
    result["verification"] = "parity-failed"
    result["all_probability_parity"] = False
    result["max_probability_error"] = 4.7e-10
    write_json(project["m3_result"], result)

    report = verifier.audit_compile_result(
        stage="M3",
        raw_root=project["raw_root"],
        dataset_hash=DATASET_HASH,
        optimizer_runs=project["singleton_runs"],
        inputs=verifier.InputRegistry(),
        repo_root=project["root"],
    )

    assert report["passed"] is False
    assert any("verification='parity-failed'" in issue for issue in report["issues"])
    assert any("probability parity failed" in issue for issue in report["issues"])
