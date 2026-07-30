from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circuitery import circuit, code_coords  # noqa: E402
from data_gen import data_generator  # noqa: E402
from problem_gen import problem_generator, representatives  # noqa: E402
from weighted_fidelity_minimization import mat_fidelities, w_fidelities  # noqa: E402

TRAIN_COUNT = 200
TEST_COUNT = 4000
TOTAL_COUNT = TRAIN_COUNT + TEST_COUNT
DATA_SEED = 30
CONTROLLED_SEEDS = (30, 31, 32, 33, 34)


@dataclass(frozen=True)
class CircleDataset:
    x: np.ndarray
    y: np.ndarray
    dataset_hash: str

    @property
    def train_x(self) -> np.ndarray:
        return self.x[:TRAIN_COUNT]

    @property
    def train_y(self) -> np.ndarray:
        return self.y[:TRAIN_COUNT]

    @property
    def test_x(self) -> np.ndarray:
        return self.x[TRAIN_COUNT:]

    @property
    def test_y(self) -> np.ndarray:
        return self.y[TRAIN_COUNT:]


@contextmanager
def preserved_numpy_random_state() -> Iterator[None]:
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _dataset_hash(x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"circle-v1|train=200|test=4000|ordered-split|")
    digest.update(np.asarray(x, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(y, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _convert_author_data(data: list[list[Any]]) -> CircleDataset:
    x = np.asarray([row[0] for row in data], dtype=np.float64)
    y = np.asarray([row[1] for row in data], dtype=np.int64)
    if x.shape != (TOTAL_COUNT, 2) or y.shape != (TOTAL_COUNT,):
        raise ValueError(f"unexpected circle dataset shapes: x={x.shape}, y={y.shape}")
    expected = (np.sum(x * x, axis=1) < (2.0 / np.pi)).astype(np.int64)
    if not np.array_equal(y, expected):
        raise ValueError("author circle labels do not match the research contract")
    return CircleDataset(x=x, y=y, dataset_hash=_dataset_hash(x, y))


def make_circle_dataset(data_seed: int = DATA_SEED) -> CircleDataset:
    """Call the authors' generator without leaking its global RNG mutation."""
    with preserved_numpy_random_state():
        np.random.seed(data_seed)
        data, _ = data_generator("circle", samples=TOTAL_COUNT)
    return _convert_author_data(data)


def legacy_circle_dataset_and_initialization(
    qubits: int,
    layers: int,
    *,
    weighted: bool,
    seed: int = DATA_SEED,
) -> tuple[CircleDataset, np.ndarray, np.ndarray, np.ndarray | None]:
    """Reproduce the authors' single RNG stream: data first, initialization next."""
    chi = "weighted_fidelity_chi" if weighted else "fidelity_chi"
    with preserved_numpy_random_state():
        np.random.seed(seed)
        data, _ = data_generator("circle", samples=TOTAL_COUNT)
        generated = problem_generator(
            "circle",
            qubits,
            layers,
            chi,
            qubits_lab=1 if weighted else qubits,
        )
    dataset = _convert_author_data(data)
    if weighted:
        theta, alpha, weights, _ = generated
        return dataset, theta, alpha, weights
    theta, alpha, _ = generated
    return dataset, theta, alpha, None


def controlled_initialization(
    qubits: int,
    layers: int,
    init_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(init_seed)
    theta = rng.rand(qubits, layers, 3)
    alpha = rng.rand(qubits, layers, 2)
    return theta, alpha


def author_statevector(
    theta: np.ndarray,
    alpha: np.ndarray,
    x: np.ndarray,
    entanglement: str,
) -> np.ndarray:
    theta_aux = code_coords(theta, alpha, np.asarray(x, dtype=float))
    psi = np.asarray(circuit(theta_aux, entanglement).psi, dtype=np.complex128)
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ValueError(f"author simulator returned non-normalized state: {norm}")
    return psi


def label_probabilities(
    theta: np.ndarray,
    alpha: np.ndarray,
    x: np.ndarray,
    entanglement: str,
) -> np.ndarray:
    psi = author_statevector(theta, alpha, x, entanglement)
    return np.asarray([abs(psi[0]) ** 2, abs(psi[-1]) ** 2], dtype=float)


def pack_parameters(theta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(theta).ravel(), np.asarray(alpha).ravel()))


def unpack_parameters(
    params: np.ndarray,
    qubits: int,
    layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    theta_size = qubits * layers * 3
    alpha_size = qubits * layers * 2
    params = np.asarray(params, dtype=float)
    if params.size != theta_size + alpha_size:
        raise ValueError(
            f"parameter count {params.size} != expected {theta_size + alpha_size}"
        )
    theta = params[:theta_size].reshape(qubits, layers, 3)
    alpha = params[theta_size:].reshape(qubits, layers, 2)
    return theta, alpha


def ordinary_objective(
    params: np.ndarray,
    *,
    qubits: int,
    layers: int,
    x: np.ndarray,
    y: np.ndarray,
    entanglement: str,
    loss_id: str,
) -> float:
    theta, alpha = unpack_parameters(params, qubits, layers)
    target_probabilities = np.empty(len(x), dtype=float)
    for index, (point, label) in enumerate(zip(x, y)):
        probs = label_probabilities(theta, alpha, point, entanglement)
        target_probabilities[index] = probs[int(label)]
    if loss_id == "legacy_amplitude":
        return -float(np.mean(np.sqrt(np.clip(target_probabilities, 0.0, 1.0))))
    if loss_id == "paper_squared":
        return float(np.mean(1.0 - target_probabilities))
    raise ValueError(f"unsupported ordinary loss: {loss_id}")


def ordinary_metrics(
    theta: np.ndarray,
    alpha: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    entanglement: str,
) -> dict[str, float]:
    scores = np.empty((len(x), 2), dtype=float)
    for index, point in enumerate(x):
        scores[index] = label_probabilities(theta, alpha, point, entanglement)
    predictions = np.argmax(scores, axis=1)
    true_scores = scores[np.arange(len(y)), y]
    other_scores = scores[np.arange(len(y)), 1 - y]
    return {
        "accuracy": float(np.mean(predictions == y)),
        "mean_true_margin": float(np.mean(true_scores - other_scores)),
        "mean_abs_decision_margin": float(np.mean(np.abs(scores[:, 0] - scores[:, 1]))),
        "min_abs_decision_margin": float(np.min(np.abs(scores[:, 0] - scores[:, 1]))),
    }


def weighted_objective(
    params: np.ndarray,
    *,
    qubits: int,
    layers: int,
    x: np.ndarray,
    y: np.ndarray,
    entanglement: str,
) -> float:
    ordinary_count = qubits * layers * 5
    theta, alpha = unpack_parameters(params[:ordinary_count], qubits, layers)
    weights = np.asarray(params[ordinary_count:], dtype=float).reshape(2, qubits)
    reprs = representatives(2, 1)
    total = 0.0
    target = np.zeros(2, dtype=float)
    for point, label in zip(x, y):
        target.fill(0.0)
        target[int(label)] = 1.0
        theta_aux = code_coords(theta, alpha, point)
        fidelities = mat_fidelities(theta_aux, weights, reprs, entanglement)
        weighted_scores = w_fidelities(fidelities, weights)
        total += 0.5 * float(np.linalg.norm(weighted_scores - target) ** 2)
    return total / len(x)


def weighted_metrics(
    theta: np.ndarray,
    alpha: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    entanglement: str,
) -> dict[str, float]:
    reprs = representatives(2, 1)
    scores = np.empty((len(x), 2), dtype=float)
    for index, point in enumerate(x):
        theta_aux = code_coords(theta, alpha, point)
        fidelities = mat_fidelities(theta_aux, weights, reprs, entanglement)
        scores[index] = w_fidelities(fidelities, weights)
    predictions = np.argmax(scores, axis=1)
    true_scores = scores[np.arange(len(y)), y]
    other_scores = scores[np.arange(len(y)), 1 - y]
    return {
        "accuracy": float(np.mean(predictions == y)),
        "mean_true_margin": float(np.mean(true_scores - other_scores)),
        "mean_abs_decision_margin": float(np.mean(np.abs(scores[:, 0] - scores[:, 1]))),
        "min_abs_decision_margin": float(np.min(np.abs(scores[:, 0] - scores[:, 1]))),
    }


def save_dataset(dataset: CircleDataset) -> Path:
    directory = REPO_ROOT / "results" / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"circle-seed-{DATA_SEED}-{dataset.dataset_hash[:16]}.npz"
    if path.exists():
        existing = np.load(path)
        existing_hash = _dataset_hash(existing["x"], existing["y"])
        if existing_hash != dataset.dataset_hash:
            raise ValueError(f"dataset artifact hash mismatch: {path}")
        return path
    np.savez_compressed(
        path,
        x=dataset.x,
        y=dataset.y,
        train_indices=np.arange(TRAIN_COUNT, dtype=np.int64),
        test_indices=np.arange(TRAIN_COUNT, TOTAL_COUNT, dtype=np.int64),
        dataset_hash=np.asarray(dataset.dataset_hash),
    )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def create_run_directory(experiment_id: str, config_id: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    leaf = f"{timestamp}-{uuid.uuid4().hex[:10]}"
    path = (
        REPO_ROOT
        / "results"
        / "raw"
        / experiment_id
        / config_id
        / f"seed-{seed}"
        / leaf
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--short")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT
    )
    untracked = run("ls-files", "--others", "--exclude-standard").splitlines()
    tracked_sources = [
        "QuantumState.py",
        "circuitery.py",
        "data_gen.py",
        "problem_gen.py",
        "fidelity_minimization.py",
        "weighted_fidelity_minimization.py",
        "src/qs_project/core.py",
        "src/qs_project/training.py",
    ]
    source_digest = hashlib.sha256()
    for relative in tracked_sources:
        path = REPO_ROOT / relative
        if path.exists():
            source_digest.update(relative.encode("utf-8"))
            source_digest.update(path.read_bytes())
    return {
        "branch": branch,
        "head": head,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked,
        "training_source_sha256": source_digest.hexdigest(),
    }


def environment_record() -> dict[str, str]:
    import qiskit
    import scipy

    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
    }


def save_checkpoint(
    path: Path,
    *,
    theta: np.ndarray,
    alpha: np.ndarray,
    weights: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "theta": np.asarray(theta, dtype=float),
        "alpha": np.asarray(alpha, dtype=float),
    }
    if weights is not None:
        payload["weights"] = np.asarray(weights, dtype=float)
    np.savez_compressed(path, **payload)


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path)
    weights = data["weights"] if "weights" in data.files else None
    return data["theta"], data["alpha"], weights

