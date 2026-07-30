#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

from qiskit import transpile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from qs_project.compile_audit import (  # noqa: E402
    TARGET_BASIS_GATES,
    bind_sample,
    bitstring_probabilities,
    build_logical_circuit,
    depth_without_measurements,
    max_probability_error,
    target_gate_counts,
)
from qs_project.core import (  # noqa: E402
    create_run_directory,
    git_revision,
    json_dump,
    load_checkpoint,
    make_circle_dataset,
    sha256_file,
)

COMPILE_SEED = 2026
SEED_TRANSPILER = 30
PARITY_TOLERANCE = 1e-10


def exact_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def latest_failed_m2_compile_result() -> Path:
    root = ROOT / "results" / "raw" / "M2" / "compile-audit" / "seed-2026"
    candidates: list[Path] = []
    for path in root.glob("*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("experiment_id") == "M2"
            and payload.get("config_id") == "compile-audit"
            and payload.get("verification") == "parity-failed"
        ):
            candidates.append(path)
    if not candidates:
        raise ValueError(f"no preserved M2 parity-failed result under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def label(probabilities: dict[str, float]) -> int:
    return 0 if probabilities.get("00", 0.0) >= probabilities.get("11", 0.0) else 1


def diagnose() -> Path:
    compile_result_path = latest_failed_m2_compile_result()
    compile_result = json.loads(compile_result_path.read_text(encoding="utf-8"))
    rows_path = Path(compile_result["rows"])
    if sha256_file(rows_path) != compile_result["rows_sha256"]:
        raise ValueError(f"M2 compile row hash mismatch: {rows_path}")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    failed = [
        row
        for row in rows
        if not row["parity_passed"]
        and row["config_id"] == "2q-l2-cz-paper_squared"
    ]
    if not failed:
        raise ValueError("preserved M2 rows contain no 2Q-CZ parity failure")
    worst = max(failed, key=lambda row: float(row["max_probability_error"]))

    checkpoint_path = Path(worst["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != worst["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    theta, alpha, weights = load_checkpoint(checkpoint_path)
    if weights is not None or theta.shape != (2, 2, 3) or alpha.shape != (2, 2, 2):
        raise ValueError("worst-row checkpoint does not match 2Q-L2 ordinary circuit")

    dataset = make_circle_dataset()
    sample_id = int(worst["sample_id"])
    point = dataset.test_x[sample_id]
    bound = bind_sample(theta, alpha, point)
    logical = build_logical_circuit(bound, entanglement="y")
    logical_probabilities = bitstring_probabilities(logical)

    level_records: list[dict[str, Any]] = []
    for level in (0, 1, 2, 3):
        compiled = transpile(
            logical,
            basis_gates=list(TARGET_BASIS_GATES),
            coupling_map=[[0, 1], [1, 0]],
            initial_layout=[0, 1],
            optimization_level=level,
            seed_transpiler=SEED_TRANSPILER,
        )
        compiled_probabilities = bitstring_probabilities(compiled)
        error = max_probability_error(logical_probabilities, compiled_probabilities)
        level_records.append(
            {
                "optimization_level": level,
                "depth_no_measure": depth_without_measurements(compiled),
                "gate_counts": target_gate_counts(compiled),
                "compiled_probabilities": compiled_probabilities,
                "max_probability_error": error,
                "probability_parity": error < PARITY_TOLERANCE,
                "logical_label": label(logical_probabilities),
                "compiled_label": label(compiled_probabilities),
                "label_parity": label(logical_probabilities)
                == label(compiled_probabilities),
            }
        )

    reproduced = (
        level_records[0]["probability_parity"]
        and level_records[1]["probability_parity"]
        and not level_records[2]["probability_parity"]
        and not level_records[3]["probability_parity"]
        and all(record["label_parity"] for record in level_records)
        and abs(
            float(level_records[3]["max_probability_error"])
            - float(worst["max_probability_error"])
        )
        < 1e-15
    )
    run_dir = create_run_directory(
        "M2", "compile-parity-diagnostic", COMPILE_SEED
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "M2",
        "config_id": "compile-parity-diagnostic",
        "compile_seed": COMPILE_SEED,
        "training_seed": int(worst["training_seed"]),
        "sample_id_test_relative": sample_id,
        "sample_id_dataset_global": sample_id + 200,
        "sample": point.tolist(),
        "label": int(dataset.test_y[sample_id]),
        "dataset_hash": dataset.dataset_hash,
        "source_compile_result": str(compile_result_path),
        "source_compile_result_sha256": sha256_file(compile_result_path),
        "source_compile_rows": str(rows_path),
        "source_compile_rows_sha256": sha256_file(rows_path),
        "source_worst_row": worst,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "seed_transpiler": SEED_TRANSPILER,
        "target_basis_gates": list(TARGET_BASIS_GATES),
        "parity_tolerance_strictly_less_than": PARITY_TOLERANCE,
        "logical_probabilities": logical_probabilities,
        "level_records": level_records,
        "conflict_reproduced": reproduced,
        "optimizer_run": False,
        "nfev": 0,
        "command": exact_command(),
        "code_revision": {
            **git_revision(),
            "diagnostic_runner_sha256": sha256_file(Path(__file__).resolve()),
            "compile_module_sha256": sha256_file(
                ROOT / "src" / "qs_project" / "compile_audit.py"
            ),
        },
        "evidence_label": "compiled-estimate",
        "verification": "conflict-reproduced" if reproduced else "diagnostic-failed",
        "raw_result_path": str(run_dir),
    }
    result_path = run_dir / "result.json"
    json_dump(result_path, payload)
    print(
        "RESULT "
        + json.dumps(
            {
                "path": str(result_path),
                "verification": payload["verification"],
                "sample_id": sample_id,
                "errors": {
                    str(record["optimization_level"]): record[
                        "max_probability_error"
                    ]
                    for record in level_records
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not reproduced:
        raise SystemExit("M2 compile parity conflict was not reproduced")
    return result_path


if __name__ == "__main__":
    diagnose()
