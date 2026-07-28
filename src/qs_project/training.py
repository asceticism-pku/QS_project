from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .core import (
    CONTROLLED_SEEDS,
    DATA_SEED,
    REPO_ROOT,
    canonical_json_hash,
    controlled_initialization,
    create_run_directory,
    environment_record,
    git_revision,
    json_dump,
    legacy_circle_dataset_and_initialization,
    make_circle_dataset,
    ordinary_metrics,
    ordinary_objective,
    pack_parameters,
    save_checkpoint,
    save_dataset,
    sha256_file,
    unpack_parameters,
    weighted_metrics,
    weighted_objective,
)

DEFAULT_OPTIONS = {
    "maxfun": 15000,
    "maxiter": 15000,
    "ftol": 2.22e-9,
    "gtol": 1e-5,
}


@dataclass(frozen=True)
class TrainingConfig:
    experiment_id: str
    config_id: str
    qubits: int
    layers: int
    entanglement: str
    loss_id: str
    rng_mode: str
    init_seed: int
    data_seed: int = DATA_SEED
    maxfun: int = 15000
    maxiter: int = 15000
    ftol: float = 2.22e-9
    gtol: float = 1e-5
    evidence_label: str = "ideal-simulation"
    run_kind: str = "optimizer"
    parent_checkpoint: str | None = None
    removed_layer: int | None = None

    @property
    def weighted(self) -> bool:
        return self.loss_id == "weighted_reduced_density"

    @property
    def controlled(self) -> bool:
        return self.rng_mode == "controlled"

    @property
    def parameter_count(self) -> int:
        count = self.qubits * self.layers * 5
        if self.weighted:
            count += 2 * self.qubits
        return count

    def optimizer_options(self) -> dict[str, Any] | None:
        if not self.controlled:
            return None
        return {
            "maxfun": self.maxfun,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "gtol": self.gtol,
        }


class ProgressObjective:
    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        progress_path: Path,
        *,
        report_every: int = 100,
    ) -> None:
        self.function = function
        self.progress_path = progress_path
        self.report_every = report_every
        self.nfev = 0
        self.best = math.inf
        self.started = time.monotonic()

    def __call__(self, params: np.ndarray) -> float:
        value = float(self.function(params))
        self.nfev += 1
        if value < self.best:
            self.best = value
        if self.nfev == 1 or self.nfev % self.report_every == 0:
            record = {
                "nfev_observed": self.nfev,
                "objective": value,
                "best_objective": self.best,
                "elapsed_seconds": time.monotonic() - self.started,
            }
            with self.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print("PROGRESS " + json.dumps(record, sort_keys=True), flush=True)
        return value


def _initial_values(
    config: TrainingConfig,
    initial_parameters: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray | None]:
    if config.rng_mode == "legacy_exact":
        if initial_parameters is not None:
            raise ValueError("legacy_exact cannot accept an external initialization")
        return legacy_circle_dataset_and_initialization(
            config.qubits,
            config.layers,
            weighted=config.weighted,
            seed=config.data_seed,
        )
    if config.rng_mode != "controlled":
        raise ValueError(f"unknown rng_mode: {config.rng_mode}")
    dataset = make_circle_dataset(config.data_seed)
    if initial_parameters is None:
        theta, alpha = controlled_initialization(
            config.qubits, config.layers, config.init_seed
        )
    else:
        theta = np.asarray(initial_parameters[0], dtype=float).copy()
        alpha = np.asarray(initial_parameters[1], dtype=float).copy()
        if theta.shape != (config.qubits, config.layers, 3):
            raise ValueError(f"initial theta shape mismatch: {theta.shape}")
        if alpha.shape != (config.qubits, config.layers, 2):
            raise ValueError(f"initial alpha shape mismatch: {alpha.shape}")
    return dataset, theta, alpha, None


def _result_payload(
    result: OptimizeResult,
    *,
    initial_objective: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "initial_fun": float(initial_objective),
        "objective_delta": float(result.fun - initial_objective),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "njev": int(result.njev) if getattr(result, "njev", None) is not None else None,
        "elapsed_seconds": float(elapsed_seconds),
    }


def run_training(
    config: TrainingConfig,
    *,
    command: str,
    initial_parameters: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    dataset, initial_theta, initial_alpha, initial_weights = _initial_values(
        config, initial_parameters
    )
    dataset_path = save_dataset(dataset)
    run_dir = create_run_directory(
        config.experiment_id, config.config_id, config.init_seed
    )
    revision = git_revision()
    config_payload = {
        **asdict(config),
        "optimizer": "L-BFGS-B",
        "optimizer_options": config.optimizer_options(),
        "dataset_hash": dataset.dataset_hash,
        "dataset_path": str(dataset_path),
        "train_count": len(dataset.train_x),
        "test_count": len(dataset.test_x),
        "parameter_count": config.parameter_count,
        "command": command,
        "code_revision": revision,
    }
    config_payload["config_fingerprint"] = canonical_json_hash(
        {
            key: value
            for key, value in config_payload.items()
            if key not in {"command", "code_revision", "dataset_path"}
        }
    )
    json_dump(run_dir / "config.json", config_payload)
    json_dump(run_dir / "environment.json", environment_record())
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    save_checkpoint(
        run_dir / "initial_checkpoint.npz",
        theta=initial_theta,
        alpha=initial_alpha,
        weights=initial_weights,
    )

    if config.weighted:
        if initial_weights is None:
            initial_weights = np.ones((2, config.qubits), dtype=float)
        x0 = np.concatenate(
            (
                pack_parameters(initial_theta, initial_alpha),
                initial_weights.ravel(),
            )
        )

        def raw_objective(params: np.ndarray) -> float:
            return weighted_objective(
                params,
                qubits=config.qubits,
                layers=config.layers,
                x=dataset.train_x,
                y=dataset.train_y,
                entanglement=config.entanglement,
            )

    else:
        x0 = pack_parameters(initial_theta, initial_alpha)

        def raw_objective(params: np.ndarray) -> float:
            return ordinary_objective(
                params,
                qubits=config.qubits,
                layers=config.layers,
                x=dataset.train_x,
                y=dataset.train_y,
                entanglement=config.entanglement,
                loss_id=config.loss_id,
            )

    initial_objective = float(raw_objective(x0))
    objective = ProgressObjective(raw_objective, run_dir / "progress.jsonl")
    started = time.monotonic()
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options=config.optimizer_options(),
    )
    elapsed = time.monotonic() - started

    ordinary_count = config.qubits * config.layers * 5
    theta, alpha = unpack_parameters(
        result.x[:ordinary_count], config.qubits, config.layers
    )
    weights = (
        np.asarray(result.x[ordinary_count:], dtype=float).reshape(2, config.qubits)
        if config.weighted
        else None
    )
    if config.weighted:
        train_metrics = weighted_metrics(
            theta,
            alpha,
            weights,
            dataset.train_x,
            dataset.train_y,
            config.entanglement,
        )
        test_metrics = weighted_metrics(
            theta,
            alpha,
            weights,
            dataset.test_x,
            dataset.test_y,
            config.entanglement,
        )
    else:
        train_metrics = ordinary_metrics(
            theta,
            alpha,
            dataset.train_x,
            dataset.train_y,
            config.entanglement,
        )
        test_metrics = ordinary_metrics(
            theta,
            alpha,
            dataset.test_x,
            dataset.test_y,
            config.entanglement,
        )

    checkpoint_path = run_dir / "checkpoint.npz"
    save_checkpoint(checkpoint_path, theta=theta, alpha=alpha, weights=weights)
    optimizer_payload = _result_payload(
        result, initial_objective=initial_objective, elapsed_seconds=elapsed
    )
    finite = all(
        np.isfinite(value)
        for value in (
            result.fun,
            train_metrics["accuracy"],
            test_metrics["accuracy"],
        )
    )
    result_payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "config_id": config.config_id,
        "seed": config.init_seed,
        "dataset_hash": dataset.dataset_hash,
        "config_fingerprint": config_payload["config_fingerprint"],
        "loss_id": config.loss_id,
        "rng_mode": config.rng_mode,
        "evidence_label": config.evidence_label,
        "verification": "artifacts-verified" if finite else "failed-nonfinite",
        "optimizer": optimizer_payload,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_checkpoint": str(run_dir / "initial_checkpoint.npz"),
        "initial_checkpoint_sha256": sha256_file(
            run_dir / "initial_checkpoint.npz"
        ),
        "dataset_artifact": str(dataset_path),
        "dataset_artifact_sha256": sha256_file(dataset_path),
        "code_revision": revision,
        "command": command,
        "raw_result_path": str(run_dir),
    }
    json_dump(run_dir / "result.json", result_payload)
    print(
        "RESULT "
        + json.dumps(
            {
                "path": str(run_dir / "result.json"),
                "config_id": config.config_id,
                "seed": config.init_seed,
                "success": optimizer_payload["success"],
                "status": optimizer_payload["status"],
                "nfev": optimizer_payload["nfev"],
                "test_accuracy": test_metrics["accuracy"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return run_dir / "result.json"


def result_files(experiment_id: str | None = None) -> list[Path]:
    root = REPO_ROOT / "results" / "raw"
    if experiment_id is not None:
        root = root / experiment_id
    return sorted(root.glob("**/result.json")) if root.exists() else []


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_verified_results(
    *,
    config_id: str,
    seeds: tuple[int, ...] = CONTROLLED_SEEDS,
    loss_id: str | None = None,
) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in result_files():
        payload = load_result(path)
        if payload.get("config_id") != config_id:
            continue
        if payload.get("seed") not in seeds:
            continue
        if loss_id is not None and payload.get("loss_id") != loss_id:
            continue
        if payload.get("verification") != "artifacts-verified":
            continue
        checkpoint = Path(payload["checkpoint"])
        if not checkpoint.exists():
            continue
        if sha256_file(checkpoint) != payload.get("checkpoint_sha256"):
            continue
        current = found.get(int(payload["seed"]))
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            found[int(payload["seed"])] = path
    return found

