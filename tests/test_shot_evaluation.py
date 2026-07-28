from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments import run_shot_evaluation as runner  # noqa: E402
from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    make_circle_dataset,
    sha256_file,
)


def test_eval_indices_artifact_is_stratified_reusable_and_contract_checked(
    tmp_path: Path,
) -> None:
    dataset = make_circle_dataset()

    path, first, payload = runner.eval_indices_artifact(
        dataset,
        root=tmp_path,
    )
    _, second, second_payload = runner.eval_indices_artifact(
        dataset,
        root=tmp_path,
    )

    assert path == tmp_path / "results" / "indices" / "eval_indices.json"
    assert payload == second_payload
    np.testing.assert_array_equal(first, second)
    assert len(first) == 1000
    assert np.unique(first).size == 1000
    assert np.count_nonzero(dataset.test_y[first] == 0) == 500
    assert np.count_nonzero(dataset.test_y[first] == 1) == 500
    assert payload["eval_seed"] == 2026
    assert payload["index_space"] == "test-set-relative"

    conflicting = dict(payload)
    conflicting["eval_seed"] = 0
    path.write_text(json.dumps(conflicting), encoding="utf-8")
    with pytest.raises(ValueError, match="conflict"):
        runner.eval_indices_artifact(dataset, root=tmp_path)


def _write_source_result(
    raw_root: Path,
    spec: runner.ModelSpec,
    seed: int,
    dataset_hash: str,
    *,
    leaf: str = "run-valid",
    recorded_hash: str | None = None,
) -> Path:
    run_dir = (
        raw_root
        / spec.source_experiment_id
        / spec.source_config_id
        / f"seed-{seed}"
        / leaf
    )
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.npz"
    np.savez_compressed(
        checkpoint,
        theta=np.zeros((1, spec.layers, 3)),
        alpha=np.zeros((1, spec.layers, 2)),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": spec.source_experiment_id,
        "config_id": spec.source_config_id,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "loss_id": "paper_squared",
        "verification": "artifacts-verified",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": recorded_hash or sha256_file(checkpoint),
        "code_revision": {"head": "test"},
    }
    result = run_dir / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_checkpoint_discovery_requires_all_ten_hash_valid_frozen_models(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    dataset_hash = "dataset-test-hash"
    for spec in runner.MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            _write_source_result(raw_root, spec, seed, dataset_hash)

    # A second, invalid result must not displace the verified checkpoint.
    _write_source_result(
        raw_root,
        runner.MODEL_SPECS[0],
        30,
        dataset_hash,
        leaf="run-invalid",
        recorded_hash="0" * 64,
    )

    records = runner.discover_checkpoint_records(
        dataset_hash,
        raw_root=raw_root,
    )

    assert len(records) == 10
    assert {(row["model_id"], row["training_seed"]) for row in records} == {
        (spec.model_id, seed)
        for spec in runner.MODEL_SPECS
        for seed in CONTROLLED_SEEDS
    }
    assert all(
        row["checkpoint_sha256"] == sha256_file(Path(row["checkpoint"]))
        for row in records
    )
    assert next(
        row
        for row in records
        if row["model_id"] == "l4-base" and row["training_seed"] == 30
    )["verified_candidate_count"] == 1


def test_checkpoint_discovery_fails_closed_when_a_seed_is_missing(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    dataset_hash = "dataset-test-hash"
    for spec in runner.MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            if spec.model_id == "l4-to-l3-pruned" and seed == 34:
                continue
            _write_source_result(raw_root, spec, seed, dataset_hash)

    with pytest.raises(
        ValueError,
        match=r"l4-to-l3-pruned, seed=34",
    ):
        runner.discover_checkpoint_records(
            dataset_hash,
            raw_root=raw_root,
        )


def test_campaign_rng_derivation_is_reproducible_and_order_independent() -> None:
    first, first_metadata = runner.campaign_rng(
        model_code=1,
        training_seed=33,
        repeat_index=7,
    )
    second, second_metadata = runner.campaign_rng(
        model_code=1,
        training_seed=33,
        repeat_index=7,
    )
    different, _ = runner.campaign_rng(
        model_code=1,
        training_seed=33,
        repeat_index=8,
    )

    first_draws = first.random(20)
    np.testing.assert_array_equal(first_draws, second.random(20))
    assert not np.array_equal(first_draws, different.random(20))
    assert first_metadata == second_metadata
    assert first_metadata["rng_bit_generator"] == "PCG64"
    assert first_metadata["rng_entropy"] == "2026:4:1:33:7"


def _fake_record(model_id: str, model_code: int, seed: int) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_code": model_code,
        "training_seed": seed,
        "checkpoint": f"/frozen/{model_id}/seed-{seed}/checkpoint.npz",
        "checkpoint_sha256": f"{model_code}{seed}".ljust(64, "0"),
    }


def test_one_campaign_emits_four_shared_rng_metric_rows() -> None:
    exact_p0 = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    labels = (exact_p0 < 0.5).astype(np.int8)
    record = _fake_record("l4-base", 0, 30)

    first = runner.run_one_campaign(
        record,
        exact_p0,
        labels,
        exact_accuracy=1.0,
        repeat_index=5,
    )
    second = runner.run_one_campaign(
        record,
        exact_p0,
        labels,
        exact_accuracy=1.0,
        repeat_index=5,
    )

    assert first == second
    assert [row["method_id"] for row in first] == [
        "fixed-128",
        "fixed-512",
        "fixed-2048",
        "adaptive",
    ]
    assert len({row["rng_entropy"] for row in first}) == 1
    assert first[0]["total_shots"] == len(labels) * 128
    assert first[1]["total_shots"] == len(labels) * 512
    assert first[2]["total_shots"] == len(labels) * 2048
    assert first[3]["cp_tail_probability"] == pytest.approx(0.05 / 6)
    assert all(row["exact_accuracy"] == 1.0 for row in first)


def test_aggregation_is_repeat_then_five_training_seeds() -> None:
    exact_p0 = np.asarray([0.2, 0.35, 0.65, 0.8])
    labels = (exact_p0 < 0.5).astype(np.int8)
    rows: list[dict[str, object]] = []
    for spec in runner.MODEL_SPECS:
        for seed in CONTROLLED_SEEDS:
            record = _fake_record(spec.model_id, spec.model_code, seed)
            for repeat_index in range(2):
                rows.extend(
                    runner.run_one_campaign(
                        record,
                        exact_p0,
                        labels,
                        exact_accuracy=1.0,
                        repeat_index=repeat_index,
                    )
                )

    metrics = runner.aggregate_campaign_metrics(
        rows,
        expected_repeats=2,
    )

    assert len(rows) == 2 * 5 * 2 * 4
    assert len(metrics["per_training_seed"]) == 2 * 5 * 4
    assert len(metrics["per_model"]) == 2 * 4
    assert len(metrics["paired_accuracy_deltas"]) == 2 * 3
    assert len(metrics["adaptive_assessment"]) == 2
    assert all(
        row["repeat_count"] == 2 for row in metrics["per_training_seed"]
    )
    assert all(
        row["training_seed_count"] == 5 for row in metrics["per_model"]
    )
    assert all(
        len(row["training_seed_delta"]) == 5
        for row in metrics["paired_accuracy_deltas"]
    )
    assert metrics["adaptive_thresholds"]["minimum_passing_training_seeds"] == 4


def test_exclusive_artifact_writers_refuse_overwrite(tmp_path: Path) -> None:
    npz_path = tmp_path / "artifact.npz"
    csv_path = tmp_path / "rows.csv"

    runner._save_npz_exclusive(npz_path, values=np.arange(3))
    runner._write_csv_exclusive(csv_path, [{"a": 1, "b": 2}])

    with pytest.raises(FileExistsError):
        runner._save_npz_exclusive(npz_path, values=np.arange(4))
    with pytest.raises(FileExistsError):
        runner._write_csv_exclusive(csv_path, [{"a": 3, "b": 4}])


def test_runner_records_all_evaluation_source_hashes() -> None:
    hashes = runner.evaluation_source_hashes()

    assert set(hashes) == {
        "src/qs_project/core.py",
        "src/qs_project/shots.py",
        "experiments/run_shot_evaluation.py",
    }
    assert all(len(value) == 64 for value in hashes.values())
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "run_training" not in source
    assert "minimize(" not in source
