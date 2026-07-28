#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from qs_project.core import (
    CONTROLLED_SEEDS,
    canonical_json_hash,
    create_run_directory,
    git_revision,
    json_dump,
    label_probabilities,
    load_checkpoint,
    make_circle_dataset,
    ordinary_objective,
    pack_parameters,
    save_dataset,
    sha256_file,
)
from qs_project.training import (
    TrainingConfig,
    find_verified_results,
    result_files,
    run_training,
)


def exact_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def ensure_training(
    config: TrainingConfig,
    *,
    initial_parameters: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    candidates = find_verified_results(
        config_id=config.config_id,
        seeds=(config.init_seed,),
        loss_id=config.loss_id,
    )
    candidate = candidates.get(config.init_seed)
    if candidate is not None:
        stored_config = json.loads(
            (candidate.parent / "config.json").read_text(encoding="utf-8")
        )
        expected = asdict(config)
        if all(stored_config.get(key) == value for key, value in expected.items()):
            print(
                "REUSE "
                + json.dumps(
                    {
                        "config_id": config.config_id,
                        "seed": config.init_seed,
                        "path": str(candidate),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return candidate
    return run_training(
        config,
        command=exact_command(),
        initial_parameters=initial_parameters,
    )


def run_loss_unit_audit() -> Path:
    dataset = make_circle_dataset()
    dataset_path = save_dataset(dataset)
    theta = np.zeros((1, 1, 3), dtype=float)
    alpha = np.zeros((1, 1, 2), dtype=float)
    x = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    y = np.asarray([0, 1], dtype=np.int64)
    params = pack_parameters(theta, alpha)
    legacy = ordinary_objective(
        params,
        qubits=1,
        layers=1,
        x=x,
        y=y,
        entanglement="n",
        loss_id="legacy_amplitude",
    )
    squared = ordinary_objective(
        params,
        qubits=1,
        layers=1,
        x=x,
        y=y,
        entanglement="n",
        loss_id="paper_squared",
    )
    probabilities = label_probabilities(theta, alpha, x[0], "n")
    passed = (
        np.allclose(probabilities, [1.0, 0.0], atol=1e-12)
        and abs(legacy + 0.5) < 1e-12
        and abs(squared - 0.5) < 1e-12
    )
    run_dir = create_run_directory("M1", "loss-unit-audit", 30)
    payload = {
        "schema_version": 1,
        "experiment_id": "M1",
        "config_id": "loss-unit-audit",
        "seed": 30,
        "dataset_hash": dataset.dataset_hash,
        "dataset_artifact": str(dataset_path),
        "legacy_amplitude": legacy,
        "paper_squared": squared,
        "state_probabilities": probabilities.tolist(),
        "passed": bool(passed),
        "command": exact_command(),
        "code_revision": git_revision(),
        "optimizer_run": False,
        "evidence_label": "ideal-simulation",
        "verification": "contract-matched" if passed else "failed",
    }
    json_dump(run_dir / "result.json", payload)
    print("RESULT " + json.dumps({"path": str(run_dir / "result.json"), **payload}))
    if not passed:
        raise SystemExit("loss unit audit failed")
    return run_dir / "result.json"


def run_p0() -> None:
    config = TrainingConfig(
        experiment_id="P0",
        config_id="circle-1q-l1-smoke",
        qubits=1,
        layers=1,
        entanglement="n",
        loss_id="paper_squared",
        rng_mode="controlled",
        init_seed=30,
        maxfun=200,
        maxiter=200,
        evidence_label="ideal-simulation",
        run_kind="smoke",
    )
    ensure_training(config)


def run_m1(selected_slice: str) -> None:
    command = exact_command()
    if selected_slice in {"all", "unit"}:
        run_loss_unit_audit()
    if selected_slice in {"all", "trend"}:
        for layers in (1, 2, 4, 8):
            config = TrainingConfig(
                experiment_id="M1",
                config_id=f"author-weighted-1q-l{layers}",
                qubits=1,
                layers=layers,
                entanglement="n",
                loss_id="weighted_reduced_density",
                rng_mode="legacy_exact",
                init_seed=30,
                evidence_label="paper-reproduction",
            )
            ensure_training(config)
    if selected_slice in {"all", "reference"}:
        config = TrainingConfig(
            experiment_id="M1",
            config_id="author-amplitude-1q-l4",
            qubits=1,
            layers=4,
            entanglement="n",
            loss_id="legacy_amplitude",
            rng_mode="legacy_exact",
            init_seed=30,
            evidence_label="paper-reproduction",
        )
        ensure_training(config)
    if selected_slice in {"all", "loss"}:
        for seed in CONTROLLED_SEEDS:
            for loss_id in ("legacy_amplitude", "paper_squared"):
                config = TrainingConfig(
                    experiment_id="M1",
                    config_id=f"1q-l4-{loss_id}",
                    qubits=1,
                    layers=4,
                    entanglement="n",
                    loss_id=loss_id,
                    rng_mode="controlled",
                    init_seed=seed,
                    evidence_label="ideal-simulation",
                )
                ensure_training(config)


def run_m2() -> None:
    baseline = find_verified_results(
        config_id="1q-l4-paper_squared", loss_id="paper_squared"
    )
    missing = sorted(set(CONTROLLED_SEEDS) - set(baseline))
    if missing:
        raise SystemExit(
            "M2 requires verified M1 1q-l4 paper_squared checkpoints; "
            f"missing seeds {missing}"
        )
    architectures = (
        ("1q-l2-paper_squared", 1, 2, "n"),
        ("2q-l2-separable-paper_squared", 2, 2, "n"),
        ("2q-l2-cz-paper_squared", 2, 2, "y"),
    )
    for seed in CONTROLLED_SEEDS:
        for config_id, qubits, layers, entanglement in architectures:
            config = TrainingConfig(
                experiment_id="M2",
                config_id=config_id,
                qubits=qubits,
                layers=layers,
                entanglement=entanglement,
                loss_id="paper_squared",
                rng_mode="controlled",
                init_seed=seed,
                evidence_label="ideal-simulation",
            )
            ensure_training(config)


def _matching_pruning_selection(
    *,
    seed: int,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    dataset_hash: str,
) -> Path | None:
    matches: list[Path] = []
    for result_path in result_files("M3"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("config_id") == "pruning-selection"
            and payload.get("seed") == seed
            and payload.get("base_checkpoint") == str(base_checkpoint)
            and payload.get("base_checkpoint_sha256") == base_checkpoint_sha256
            and payload.get("dataset_hash") == dataset_hash
            and payload.get("verification") == "artifacts-verified"
        ):
            matches.append(result_path)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _pruning_selection(
    *,
    seed: int,
    base_result_path: Path,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    theta: np.ndarray,
    alpha: np.ndarray,
) -> tuple[int, Path]:
    dataset = make_circle_dataset()
    existing = _matching_pruning_selection(
        seed=seed,
        base_checkpoint=base_checkpoint,
        base_checkpoint_sha256=base_checkpoint_sha256,
        dataset_hash=dataset.dataset_hash,
    )
    if existing is not None:
        payload = json.loads(existing.read_text(encoding="utf-8"))
        selected = int(payload["selected_layer_zero_based"])
        print(
            "REUSE "
            + json.dumps(
                {
                    "config_id": "pruning-selection",
                    "seed": seed,
                    "selected_layer_zero_based": selected,
                    "path": str(existing),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return selected, existing

    if theta.shape != (1, 4, 3) or alpha.shape != (1, 4, 2):
        raise ValueError(
            f"M3 base checkpoint has unexpected shapes theta={theta.shape}, "
            f"alpha={alpha.shape}"
        )
    base_loss = ordinary_objective(
        pack_parameters(theta, alpha),
        qubits=1,
        layers=4,
        x=dataset.train_x,
        y=dataset.train_y,
        entanglement="n",
        loss_id="paper_squared",
    )
    candidates: list[dict[str, float | int]] = []
    for layer in range(4):
        pruned_theta = np.delete(theta, layer, axis=1)
        pruned_alpha = np.delete(alpha, layer, axis=1)
        loss = ordinary_objective(
            pack_parameters(pruned_theta, pruned_alpha),
            qubits=1,
            layers=3,
            x=dataset.train_x,
            y=dataset.train_y,
            entanglement="n",
            loss_id="paper_squared",
        )
        candidates.append(
            {
                "layer_zero_based": layer,
                "layer_one_based": layer + 1,
                "unfinetuned_train_loss": float(loss),
                "loss_increase_from_base": float(loss - base_loss),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["loss_increase_from_base"]),
            int(item["layer_zero_based"]),
        ),
    )
    selected_layer = int(selected["layer_zero_based"])
    dataset_path = save_dataset(dataset)
    run_dir = create_run_directory("M3", "pruning-selection", seed)
    payload = {
        "schema_version": 1,
        "experiment_id": "M3",
        "config_id": "pruning-selection",
        "seed": seed,
        "selection_rule": (
            "minimum unfinetuned paper_squared training-loss increase; "
            "ties use lower zero-based layer index"
        ),
        "base_result": str(base_result_path),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_train_loss_recomputed": float(base_loss),
        "candidates": candidates,
        "selected_layer_zero_based": selected_layer,
        "selected_layer_one_based": selected_layer + 1,
        "dataset_hash": dataset.dataset_hash,
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "command": exact_command(),
        "code_revision": git_revision(),
        "selection_fingerprint": canonical_json_hash(
            {
                "seed": seed,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "dataset_hash": dataset.dataset_hash,
                "candidates": candidates,
                "selected_layer_zero_based": selected_layer,
            }
        ),
        "optimizer_run": False,
        "evidence_label": "ideal-simulation",
        "verification": "artifacts-verified",
        "raw_result_path": str(run_dir),
    }
    json_dump(run_dir / "result.json", payload)
    print(
        "RESULT "
        + json.dumps(
            {
                "path": str(run_dir / "result.json"),
                "config_id": "pruning-selection",
                "seed": seed,
                "selected_layer_zero_based": selected_layer,
                "base_train_loss": base_loss,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return selected_layer, run_dir / "result.json"


def run_m3() -> None:
    baseline = find_verified_results(
        config_id="1q-l4-paper_squared", loss_id="paper_squared"
    )
    missing = sorted(set(CONTROLLED_SEEDS) - set(baseline))
    if missing:
        raise SystemExit(
            "M3 requires verified M1 1q-l4 paper_squared checkpoints; "
            f"missing seeds {missing}"
        )

    for seed in CONTROLLED_SEEDS:
        base_result_path = baseline[seed]
        base_result = json.loads(base_result_path.read_text(encoding="utf-8"))
        base_checkpoint = Path(base_result["checkpoint"])
        expected_sha256 = str(base_result["checkpoint_sha256"])
        actual_sha256 = sha256_file(base_checkpoint)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"M3 base checkpoint hash mismatch for seed {seed}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        theta, alpha, weights = load_checkpoint(base_checkpoint)
        if weights is not None:
            raise SystemExit(f"M3 seed {seed} unexpectedly has weighted checkpoint")

        selected_layer, _ = _pruning_selection(
            seed=seed,
            base_result_path=base_result_path,
            base_checkpoint=base_checkpoint,
            base_checkpoint_sha256=actual_sha256,
            theta=theta,
            alpha=alpha,
        )
        parent = str(base_checkpoint)
        configurations = (
            (
                "l4-to-l3-pruned",
                selected_layer,
                np.delete(theta, selected_layer, axis=1),
                np.delete(alpha, selected_layer, axis=1),
            ),
            (
                "l4-truncate-last",
                3,
                np.delete(theta, 3, axis=1),
                np.delete(alpha, 3, axis=1),
            ),
        )
        for config_id, removed_layer, initial_theta, initial_alpha in configurations:
            config = TrainingConfig(
                experiment_id="M3",
                config_id=config_id,
                qubits=1,
                layers=3,
                entanglement="n",
                loss_id="paper_squared",
                rng_mode="controlled",
                init_seed=seed,
                evidence_label="ideal-simulation",
                parent_checkpoint=parent,
                removed_layer=removed_layer,
            )
            ensure_training(
                config,
                initial_parameters=(initial_theta, initial_alpha),
            )

        scratch = TrainingConfig(
            experiment_id="M3",
            config_id="l3-scratch",
            qubits=1,
            layers=3,
            entanglement="n",
            loss_id="paper_squared",
            rng_mode="controlled",
            init_seed=seed,
            maxfun=30000,
            maxiter=30000,
            evidence_label="ideal-simulation",
        )
        ensure_training(scratch)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("p0")
    m1 = subparsers.add_parser("m1")
    m1.add_argument(
        "--slice",
        choices=("all", "unit", "trend", "reference", "loss"),
        default="all",
    )
    subparsers.add_parser("m2")
    subparsers.add_parser("m3")
    args = parser.parse_args()
    if args.stage == "p0":
        run_p0()
    elif args.stage == "m1":
        run_m1(args.slice)
    elif args.stage == "m2":
        run_m2()
    elif args.stage == "m3":
        run_m3()


if __name__ == "__main__":
    main()
