from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_compile_audit as runner
from qs_project.core import sha256_file


def test_m3_checkpoint_contract_reuses_m1_base() -> None:
    specs = runner.checkpoint_specs("m3")

    assert [
        (
            spec.model_id,
            spec.source_config_id,
            spec.qubits,
            spec.layers,
            spec.entanglement,
            spec.source_stage,
        )
        for spec in specs
    ] == [
        ("l4-base", "1q-l4-paper_squared", 1, 4, "n", "M1"),
        ("l4-to-l3-pruned", "l4-to-l3-pruned", 1, 3, "n", "M3"),
        ("l4-truncate-last", "l4-truncate-last", 1, 3, "n", "M3"),
        ("l3-scratch", "l3-scratch", 1, 3, "n", "M3"),
    ]


def test_missing_m3_checkpoint_preserves_failure_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "raw-run"
    indices_path = tmp_path / "compile_indices.json"
    indices_path.write_text("{}\n", encoding="utf-8")
    dataset = SimpleNamespace(
        dataset_hash="dataset-hash",
        test_x=np.zeros((2, 2)),
    )

    def create_run_directory(*_args):
        run_dir.mkdir()
        return run_dir

    monkeypatch.setattr(runner, "create_run_directory", create_run_directory)
    monkeypatch.setattr(runner, "make_circle_dataset", lambda: dataset)
    monkeypatch.setattr(
        runner,
        "compile_indices_artifact",
        lambda **_kwargs: (
            indices_path,
            np.asarray([0, 1]),
            {"indices_content_sha256": "content-hash"},
        ),
    )
    monkeypatch.setattr(runner, "find_verified_results", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "_revision_record", lambda: {"head": "test"})

    with pytest.raises(ValueError, match="missing verified seeds"):
        runner.run_m3_compile_audit()

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["verification"] == "failed-exception"
    assert result["failure"]["type"] == "ValueError"
    assert result["expected_checkpoint_count"] == 20
    assert result["checkpoint_count"] == 0
    assert result["row_count"] == 0
    assert (run_dir / "compile_rows.jsonl").read_text(encoding="utf-8") == ""
    assert (run_dir / "checkpoint_records.json").is_file()
    assert (run_dir / "compile_summary.json").is_file()


def test_parity_failure_preserves_rows_and_exits_nonzero(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "raw-run"
    indices_path = tmp_path / "compile_indices.json"
    indices_path.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.npz"
    np.savez_compressed(
        checkpoint_path,
        theta=np.zeros((1, 4, 3)),
        alpha=np.zeros((1, 4, 2)),
    )
    source_result_path = tmp_path / "source-result.json"
    source_result_path.write_text(
        json.dumps(
            {
                "experiment_id": "M1",
                "config_id": "1q-l4-paper_squared",
                "seed": 30,
                "loss_id": "paper_squared",
                "dataset_hash": "dataset-hash",
                "verification": "artifacts-verified",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        dataset_hash="dataset-hash",
        test_x=np.zeros((2, 2)),
    )
    spec = runner.CheckpointSpec(
        "l4-base",
        "1q-l4-paper_squared",
        1,
        4,
        "n",
        "M1",
    )

    def create_run_directory(*_args):
        run_dir.mkdir()
        return run_dir

    def fake_audit_samples(*_args, sample_ids, **_kwargs):
        rows = []
        for sample_id in sample_ids:
            for level in (0, 3):
                failed = sample_id == 1 and level == 3
                rows.append(
                    {
                        "sample_id": sample_id,
                        "optimization_level": level,
                        "depth_no_measure": 12 - level,
                        "rz_count": 8,
                        "sx_count": 4,
                        "x_count": 0,
                        "cz_count": 0,
                        "max_probability_error": 2e-10 if failed else 0.0,
                        "parity_passed": not failed,
                        "logical_probabilities": {"0": 0.6, "1": 0.4},
                        "compiled_probabilities": {"0": 0.6, "1": 0.4},
                    }
                )
        return rows

    monkeypatch.setattr(runner, "CONTROLLED_SEEDS", (30,))
    monkeypatch.setattr(runner, "checkpoint_specs", lambda _stage: (spec,))
    monkeypatch.setattr(runner, "create_run_directory", create_run_directory)
    monkeypatch.setattr(runner, "make_circle_dataset", lambda: dataset)
    monkeypatch.setattr(
        runner,
        "compile_indices_artifact",
        lambda **_kwargs: (
            indices_path,
            np.asarray([0, 1]),
            {"indices_content_sha256": "content-hash"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "find_verified_results",
        lambda **_kwargs: {30: source_result_path},
    )
    monkeypatch.setattr(runner, "audit_samples", fake_audit_samples)
    monkeypatch.setattr(runner, "_revision_record", lambda: {"head": "test"})

    with pytest.raises(SystemExit, match="raw artifacts were preserved"):
        runner.run_m3_compile_audit()

    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (run_dir / "compile_summary.json").read_text(encoding="utf-8")
    )
    assert result["verification"] == "parity-failed"
    assert result["checkpoint_count"] == 1
    assert result["point_count_per_checkpoint"] == 2
    assert result["row_count"] == 4
    assert result["all_probability_parity"] is False
    assert result["all_label_parity"] is True
    assert [row["config_id"] for row in summary] == ["l4-base", "l4-base"]
    assert [row["optimization_level"] for row in summary] == [0, 3]
    assert sum(1 for _ in (run_dir / "compile_rows.jsonl").open()) == 4
