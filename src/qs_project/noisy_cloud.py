"""Noisy and Origin Quantum Cloud evaluation for appendix classifiers.

The appendix classifiers are frozen before this module is used.  Ordinary
fidelity classifiers are read from computational-basis probabilities whenever
their label states are basis states.  Weighted-fidelity classifiers are read
with X/Y/Z tomography, matching the reduced-density-matrix definition used by
the original code.

``pyqpanda3`` is an optional dependency.  Local noisy evaluation uses only
NumPy, while :class:`OriginCloudBackend` imports the SDK lazily.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


REPO_ROOT = Path(__file__).resolve().parents[2]
APPENDIX_SEED = 30
PROBLEM_CLASSES = {
    "non convex": 2,
    "crown": 2,
    "sphere": 2,
    "squares": 4,
    "wavy lines": 4,
}

_I = np.eye(2, dtype=np.complex128)
_X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SDG = np.asarray([[1, 0], [0, -1j]], dtype=np.complex128)
_PAULIS = (_I, _X, _Y, _Z)


@dataclass(frozen=True)
class AppendixModel:
    """One frozen appendix result and its classification metadata."""

    chi: str
    problem: str
    qubits: int
    layers: int
    entanglement: str
    theta: NDArray[np.float64]
    alpha: NDArray[np.float64]
    weights: NDArray[np.float64] | None
    representatives: tuple[NDArray[np.complex128], ...]
    recorded_accuracy: float
    summary_path: Path

    @property
    def classes(self) -> int:
        return len(self.representatives)

    @property
    def weighted(self) -> bool:
        return self.chi == "weighted_fidelity_chi"

    @property
    def measurement_bases(self) -> tuple[str, ...]:
        if self.weighted:
            return ("x", "y", "z")
        if all(_basis_state_index(state) is not None for state in self.representatives):
            return ("z",)
        if self.qubits == 1:
            return ("x", "y", "z")
        raise ValueError("non-basis multi-qubit labels require full tomography")


@dataclass(frozen=True)
class DepolarizingNoise:
    """Gate and symmetric readout error probabilities."""

    single_qubit: float = 0.001
    two_qubit: float = 0.01
    readout: float = 0.002

    def __post_init__(self) -> None:
        for name, value in (
            ("single_qubit", self.single_qubit),
            ("two_qubit", self.two_qubit),
            ("readout", self.readout),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


def _field(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"missing summary field matching {pattern!r}")
    return match.group(1).strip()


def _weights_from_summary(
    text: str,
    classes: int,
    qubits: int,
) -> NDArray[np.float64]:
    match = re.search(
        r"^WEIGHTS\s*=\s*\n(.*?)\nchi\*\*2\s*=",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("weighted summary has no WEIGHTS block")
    values = np.fromstring(
        match.group(1).replace("[", " ").replace("]", " "),
        sep=" ",
        dtype=float,
    )
    if values.size != classes * qubits:
        raise ValueError(
            f"weight count {values.size} != expected {classes * qubits}"
        )
    return values.reshape(classes, qubits)


def _entanglement_from_path(path: Path, qubits: int) -> str:
    if (
        qubits > 1
        and "entangled" in path.parts
        and "not_entangled" not in path.parts
    ):
        return "y"
    return "n"


def load_appendix_model(
    *,
    chi: str,
    problem: str,
    qubits: int,
    layers: int,
    entanglement: str,
    seed: int = APPENDIX_SEED,
    root: Path = REPO_ROOT,
) -> AppendixModel:
    """Locate and load one frozen result produced by the appendix campaign."""

    if chi not in {"fidelity_chi", "weighted_fidelity_chi"}:
        raise ValueError("chi must be fidelity_chi or weighted_fidelity_chi")
    if problem not in PROBLEM_CLASSES:
        raise ValueError(f"unsupported appendix problem: {problem!r}")
    ent = entanglement.strip().lower()[0]
    if ent not in {"y", "n"}:
        raise ValueError("entanglement must be y or n")

    filename = f"appendix_seed_{seed}_summary.txt"
    matches: list[tuple[Path, str]] = []
    for summary in (root / chi / problem).rglob(filename):
        text = summary.read_text(encoding="utf-8")
        if (
            int(_field(r"^Number of qubits = (\d+)$", text)) == qubits
            and int(_field(r"^Number of layers = (\d+)$", text)) == layers
            and _entanglement_from_path(summary, qubits) == ent
        ):
            matches.append((summary, text))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one appendix result for "
            f"{chi}/{problem}/{qubits}q/{layers}L/entanglement={ent}; "
            f"found {len(matches)}"
        )

    summary, text = matches[0]
    theta_values = np.loadtxt(
        summary.with_name(summary.name.replace("_summary.txt", "_theta.txt"))
    )
    alpha_values = np.loadtxt(
        summary.with_name(summary.name.replace("_summary.txt", "_alpha.txt"))
    )
    theta = np.asarray(theta_values, dtype=float).reshape(qubits, layers, 3)
    dimension = int(alpha_values.size // (qubits * layers))
    alpha = np.asarray(alpha_values, dtype=float).reshape(qubits, layers, dimension)
    classes = PROBLEM_CLASSES[problem]

    from problem_gen import representatives

    label_qubits = 1 if chi == "weighted_fidelity_chi" else qubits
    reprs = tuple(
        np.asarray(state, dtype=np.complex128)
        for state in representatives(classes, label_qubits)
    )
    weights = (
        _weights_from_summary(text, classes, qubits)
        if chi == "weighted_fidelity_chi"
        else None
    )
    return AppendixModel(
        chi=chi,
        problem=problem,
        qubits=qubits,
        layers=layers,
        entanglement=ent,
        theta=theta,
        alpha=alpha,
        weights=weights,
        representatives=reprs,
        recorded_accuracy=float(_field(r"^acc_test = ([0-9.eE+-]+)$", text)),
        summary_path=summary,
    )


def appendix_test_data(
    problem: str,
    *,
    seed: int = APPENDIX_SEED,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Regenerate the exact ordered appendix test split."""

    if problem not in PROBLEM_CLASSES:
        raise ValueError(f"unsupported appendix problem: {problem!r}")
    from data_gen import data_generator

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        data, _ = data_generator(problem)
    finally:
        np.random.set_state(state)
    split = 500 if problem == "sphere" else 200
    test = data[split:]
    return (
        np.asarray([row[0] for row in test], dtype=float),
        np.asarray([row[1] for row in test], dtype=np.int64),
    )


def stratified_subset(
    x: ArrayLike,
    y: ArrayLike,
    points: int,
    *,
    seed: int = 2026,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Select an approximately balanced, reproducible evaluation subset."""

    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=np.int64)
    if points <= 0 or points > len(labels):
        raise ValueError(f"points must lie in [1, {len(labels)}]")
    if points == len(labels):
        indices = np.arange(len(labels), dtype=np.int64)
        return features.copy(), labels.copy(), indices
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(points, len(classes))
    chosen: list[NDArray[np.int64]] = []
    for offset, label in enumerate(classes):
        count = base + int(offset < remainder)
        candidates = np.flatnonzero(labels == label)
        if count > candidates.size:
            raise ValueError(f"class {label} has only {candidates.size} points")
        chosen.append(rng.choice(candidates, size=count, replace=False))
    indices = np.sort(np.concatenate(chosen)).astype(np.int64)
    return features[indices], labels[indices], indices


def bind_coordinates(model: AppendixModel, x: ArrayLike) -> NDArray[np.float64]:
    point = np.asarray(x, dtype=float)
    if point.ndim != 1 or point.size != model.alpha.shape[2]:
        raise ValueError(
            f"sample must have shape ({model.alpha.shape[2]},), got {point.shape}"
        )
    bound = model.theta.copy()
    bound[:, :, : point.size] += (
        model.alpha * point[np.newaxis, np.newaxis, :]
    )
    return bound


def author_u3(angles: ArrayLike) -> NDArray[np.complex128]:
    """Return the exact U3 convention implemented in ``QuantumState.py``."""

    theta, phi, lamb = np.asarray(angles, dtype=float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ep, el = np.exp(0.5j * phi), np.exp(0.5j * lamb)
    return np.asarray(
        [
            [c * ep * el, -s * ep * np.conj(el)],
            [s * np.conj(ep) * el, c * np.conj(ep) * np.conj(el)],
        ],
        dtype=np.complex128,
    )


def _expand_single(
    gate: NDArray[np.complex128],
    qubit: int,
    qubits: int,
) -> NDArray[np.complex128]:
    factors = [gate if index == qubit else _I for index in range(qubits - 1, -1, -1)]
    full = factors[0]
    for factor in factors[1:]:
        full = np.kron(full, factor)
    return full


def _apply_unitary(
    rho: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    return unitary @ rho @ np.conj(unitary.T)


def _single_depolarize(
    rho: NDArray[np.complex128],
    qubit: int,
    qubits: int,
    probability: float,
) -> NDArray[np.complex128]:
    if probability == 0.0:
        return rho
    mixed = (1.0 - probability) * rho
    for pauli in (_X, _Y, _Z):
        full = _expand_single(pauli, qubit, qubits)
        mixed += (probability / 3.0) * _apply_unitary(rho, full)
    return mixed


def _two_depolarize(
    rho: NDArray[np.complex128],
    left: int,
    right: int,
    qubits: int,
    probability: float,
) -> NDArray[np.complex128]:
    if probability == 0.0:
        return rho
    mixed = (1.0 - probability) * rho
    for left_pauli in _PAULIS:
        for right_pauli in _PAULIS:
            if left_pauli is _I and right_pauli is _I:
                continue
            full = _expand_single(left_pauli, left, qubits)
            full = full @ _expand_single(right_pauli, right, qubits)
            mixed += (probability / 15.0) * _apply_unitary(rho, full)
    return mixed


def _cz_unitary(left: int, right: int, qubits: int) -> NDArray[np.complex128]:
    diagonal = np.ones(2**qubits, dtype=np.complex128)
    for index in range(diagonal.size):
        if ((index >> left) & 1) and ((index >> right) & 1):
            diagonal[index] = -1.0
    return np.diag(diagonal)


def _entangling_pairs(qubits: int, layer: int) -> tuple[tuple[int, int], ...]:
    if qubits == 2:
        return ((0, 1),)
    if qubits == 4 and layer % 2 == 0:
        return ((0, 1), (2, 3))
    if qubits == 4:
        return ((1, 2), (0, 3))
    return ()


def simulate_density_matrix(
    theta_aux: ArrayLike,
    entanglement: str,
    noise: DepolarizingNoise,
) -> NDArray[np.complex128]:
    """Evolve the author's circuit with gate-local depolarizing channels."""

    values = np.asarray(theta_aux, dtype=float)
    if values.ndim != 3 or values.shape[0] not in (1, 2, 4):
        raise ValueError("theta_aux must describe a 1, 2, or 4 qubit circuit")
    if values.shape[2] not in (3, 6):
        raise ValueError("each layer must contain 3 or 6 author U3 parameters")
    qubits, layers, parameters = values.shape
    rho = np.zeros((2**qubits, 2**qubits), dtype=np.complex128)
    rho[0, 0] = 1.0

    for layer in range(layers):
        for qubit in range(qubits):
            for offset in range(0, parameters, 3):
                unitary = _expand_single(
                    author_u3(values[qubit, layer, offset : offset + 3]),
                    qubit,
                    qubits,
                )
                rho = _apply_unitary(rho, unitary)
                rho = _single_depolarize(
                    rho, qubit, qubits, noise.single_qubit
                )
        if entanglement.lower()[0] == "y" and layer < layers - 1:
            for left, right in _entangling_pairs(qubits, layer):
                rho = _apply_unitary(rho, _cz_unitary(left, right, qubits))
                rho = _two_depolarize(
                    rho, left, right, qubits, noise.two_qubit
                )
    return rho


def _readout_noise(
    probabilities: NDArray[np.float64],
    qubits: int,
    probability: float,
) -> NDArray[np.float64]:
    if probability == 0.0:
        return probabilities
    noisy = np.zeros_like(probabilities)
    for source, source_probability in enumerate(probabilities):
        for mask in range(2**qubits):
            flips = mask.bit_count()
            transition = probability**flips * (1.0 - probability) ** (qubits - flips)
            noisy[source ^ mask] += source_probability * transition
    return noisy


def measurement_probabilities(
    rho: ArrayLike,
    basis: str,
    *,
    readout: float = 0.0,
) -> NDArray[np.float64]:
    """Return noisy computational probabilities after a global X/Y/Z readout."""

    density = np.asarray(rho, dtype=np.complex128)
    qubits = int(np.log2(density.shape[0]))
    if density.shape != (2**qubits, 2**qubits):
        raise ValueError("rho must be a square power-of-two density matrix")
    basis = basis.lower()
    if basis not in {"x", "y", "z"}:
        raise ValueError("basis must be x, y, or z")
    rotated = density
    if basis != "z":
        single = _H if basis == "x" else _H @ _SDG
        full = _expand_single(single, 0, qubits)
        for qubit in range(1, qubits):
            full = full @ _expand_single(single, qubit, qubits)
        rotated = _apply_unitary(rotated, full)
    probabilities = np.real(np.diag(rotated))
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= np.sum(probabilities)
    return _readout_noise(probabilities, qubits, readout)


def sample_probabilities(
    probabilities: ArrayLike,
    shots: int | None,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, None)
    values /= np.sum(values)
    if shots is None:
        return values
    if shots <= 0:
        raise ValueError("shots must be positive or None")
    return rng.multinomial(shots, values).astype(float) / shots


def _basis_state_index(state: ArrayLike, tolerance: float = 1e-10) -> int | None:
    vector = np.asarray(state, dtype=np.complex128)
    probabilities = np.abs(vector) ** 2
    index = int(np.argmax(probabilities))
    expected = np.zeros_like(probabilities)
    expected[index] = 1.0
    return index if np.max(np.abs(probabilities - expected)) <= tolerance else None


def _expectation_from_probabilities(
    probabilities: NDArray[np.float64],
    qubit: int,
) -> float:
    signs = np.asarray(
        [1.0 if ((index >> qubit) & 1) == 0 else -1.0 for index in range(len(probabilities))]
    )
    return float(np.dot(signs, probabilities))


def scores_from_measurements(
    model: AppendixModel,
    measured: Mapping[str, ArrayLike],
) -> NDArray[np.float64]:
    """Apply the original fidelity or weighted-fidelity decision rule."""

    probabilities = {
        basis: np.asarray(values, dtype=float) for basis, values in measured.items()
    }
    basis_indices = [_basis_state_index(state) for state in model.representatives]
    if not model.weighted and all(index is not None for index in basis_indices):
        z = probabilities["z"]
        return np.asarray([z[int(index)] for index in basis_indices], dtype=float)

    bloch = np.empty((model.qubits, 3), dtype=float)
    for qubit in range(model.qubits):
        bloch[qubit] = [
            _expectation_from_probabilities(probabilities["x"], qubit),
            _expectation_from_probabilities(probabilities["y"], qubit),
            _expectation_from_probabilities(probabilities["z"], qubit),
        ]
    label_bloch = np.asarray(
        [
            [
                2.0 * np.real(np.conj(state[0]) * state[1]),
                2.0 * np.imag(np.conj(state[0]) * state[1]),
                abs(state[0]) ** 2 - abs(state[1]) ** 2,
            ]
            for state in model.representatives
        ],
        dtype=float,
    )
    fidelities = 0.5 * (
        1.0 + label_bloch[:, np.newaxis, :] @ bloch[:, :, np.newaxis]
    )[:, :, 0]
    if model.weighted:
        if model.weights is None:
            raise ValueError("weighted model has no weights")
        return np.sum(fidelities * model.weights, axis=1)
    if model.qubits != 1:
        raise ValueError("non-basis ordinary tomography is only supported for 1q")
    return fidelities[:, 0]


def evaluate_local(
    model: AppendixModel,
    x: ArrayLike,
    y: ArrayLike,
    *,
    noise: DepolarizingNoise,
    shots: int | None,
    seed: int = 2026,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one frozen model with local gate noise and finite sampling."""

    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    predictions: list[int] = []
    margins: list[float] = []
    for index, (point, label) in enumerate(zip(features, labels)):
        theta_aux = bind_coordinates(model, point)
        rho = simulate_density_matrix(theta_aux, model.entanglement, noise)
        measured = {
            basis: sample_probabilities(
                measurement_probabilities(
                    rho, basis, readout=noise.readout
                ),
                shots,
                rng,
            )
            for basis in model.measurement_bases
        }
        scores = scores_from_measurements(model, measured)
        prediction = int(np.argmax(scores))
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2])
        predictions.append(prediction)
        margins.append(margin)
        row: dict[str, Any] = {
            "point": index,
            "label": int(label),
            "prediction": prediction,
            "correct": int(prediction == label),
            "margin": margin,
        }
        for class_id, score in enumerate(scores):
            row[f"score_{class_id}"] = float(score)
        rows.append(row)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    summary = {
        "backend": "local-density-matrix",
        "points": int(len(labels)),
        "shots_per_basis": shots,
        "measurement_bases": list(model.measurement_bases),
        "circuits_executed": int(len(labels) * len(model.measurement_bases)),
        "accuracy": float(np.mean(prediction_array == labels)),
        "mean_decision_margin": float(np.mean(margins)),
        "single_qubit_error": noise.single_qubit,
        "two_qubit_error": noise.two_qubit,
        "readout_error": noise.readout,
        "rng_seed": seed,
    }
    return summary, rows


def _normalize_cloud_probabilities(
    raw: Mapping[Any, Any] | Sequence[float],
    qubits: int,
) -> NDArray[np.float64]:
    if isinstance(raw, Mapping):
        probabilities = np.zeros(2**qubits, dtype=float)
        for key, value in raw.items():
            if isinstance(key, str):
                normalized = key.replace(" ", "").replace("0b", "")
                index = (
                    int(normalized, 16)
                    if normalized.lower().startswith("0x")
                    else int(normalized, 2)
                )
            else:
                index = int(key)
            probabilities[index] = float(value)
    else:
        probabilities = np.asarray(raw, dtype=float)
    if probabilities.shape != (2**qubits,):
        raise ValueError(
            f"cloud result has shape {probabilities.shape}; "
            f"expected {(2**qubits,)}"
        )
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("cloud probabilities sum to zero")
    return probabilities / total


class OriginCloudBackend:
    """Thin pyqpanda3 adapter with lazy authentication and job-id capture."""

    def __init__(
        self,
        backend_name: str,
        *,
        api_key: str | None = None,
    ) -> None:
        try:
            from pyqpanda3.qcloud import QCloudService
        except ImportError as exc:
            raise RuntimeError(
                "Origin Cloud requires the optional dependency "
                "`pip install pyqpanda3==0.4.0`"
            ) from exc
        token = api_key or os.environ.get("QPANDA_QCLOUD_API_KEY")
        if not token:
            raise RuntimeError(
                "set QPANDA_QCLOUD_API_KEY to the API key from Origin Cloud"
            )
        self._backend_name = backend_name
        self._service = QCloudService(token)
        backends = self._service.backends()
        if backend_name not in backends:
            raise RuntimeError(
                f"Origin Cloud account does not expose backend {backend_name!r}; "
                "run --list-origin-backends"
            )
        if not backends[backend_name]:
            raise RuntimeError(
                f"Origin Cloud backend {backend_name!r} is currently unavailable"
            )
        self._backend = self._service.backend(backend_name)

    @staticmethod
    def available_backends(api_key: str | None = None) -> Any:
        try:
            from pyqpanda3.qcloud import QCloudService
        except ImportError as exc:
            raise RuntimeError(
                "Origin Cloud requires `pip install pyqpanda3==0.4.0`"
            ) from exc
        token = api_key or os.environ.get("QPANDA_QCLOUD_API_KEY")
        if not token:
            raise RuntimeError("set QPANDA_QCLOUD_API_KEY first")
        return QCloudService(token).backends()

    @staticmethod
    def _append_program(
        model: AppendixModel,
        theta_aux: NDArray[np.float64],
        basis: str,
    ) -> Any:
        from pyqpanda3.core import CZ, H, RZ, U3, QProg, measure

        prog = QProg()
        for layer in range(model.layers):
            for qubit in range(model.qubits):
                angles = theta_aux[qubit, layer]
                prog << U3(
                    qubit,
                    float(angles[0]),
                    -float(angles[1]),
                    -float(angles[2]),
                )
            if model.entanglement == "y" and layer < model.layers - 1:
                for left, right in _entangling_pairs(model.qubits, layer):
                    prog << CZ(left, right)
        for qubit in range(model.qubits):
            if basis == "x":
                prog << H(qubit)
            elif basis == "y":
                prog << RZ(qubit, -float(np.pi / 2)) << H(qubit)
            prog << measure(qubit, qubit)
        return prog

    @staticmethod
    def _job_id(job: Any) -> str:
        value = getattr(job, "job_id", "")
        return str(value() if callable(value) else value)

    def evaluate(
        self,
        model: AppendixModel,
        x: ArrayLike,
        y: ArrayLike,
        *,
        shots: int,
        noise: DepolarizingNoise,
        batch_size: int = 20,
        on_job: Any | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Submit batched tomography circuits and return classifier metrics."""

        if shots <= 0:
            raise ValueError("shots must be positive")
        if noise.readout != 0.0:
            raise ValueError(
                "Origin Cloud QCloudNoiseModel does not expose readout noise; "
                "set --readout-multiplier 0"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            from pyqpanda3.qcloud import DataBase, NOISE_MODEL, QCloudNoiseModel
        except ImportError as exc:
            raise RuntimeError("pyqpanda3 qcloud module is unavailable") from exc

        features = np.asarray(x, dtype=float)
        labels = np.asarray(y, dtype=np.int64)
        bases = model.measurement_bases
        programs = [
            self._append_program(model, bind_coordinates(model, point), basis)
            for point in features
            for basis in bases
        ]
        model_noise = QCloudNoiseModel(
            NOISE_MODEL.DEPOLARIZING_KRAUS_OPERATOR,
            [noise.single_qubit],
            [noise.two_qubit],
        )
        all_probabilities: list[NDArray[np.float64]] = []
        job_ids: list[str] = []
        for start in range(0, len(programs), batch_size):
            batch = programs[start : start + batch_size]
            job = self._backend.run(batch, shots=shots, model=model_noise)
            job_id = self._job_id(job)
            job_ids.append(job_id)
            if on_job is not None:
                on_job(job_id, start, len(batch))
            result = job.result()
            raw_list = result.get_probs_list(base=DataBase.Binary)
            if len(raw_list) != len(batch):
                raise RuntimeError("Origin Cloud returned an incomplete batch")
            all_probabilities.extend(
                _normalize_cloud_probabilities(raw, model.qubits)
                for raw in raw_list
            )

        rows: list[dict[str, Any]] = []
        predictions: list[int] = []
        margins: list[float] = []
        for index, label in enumerate(labels):
            offset = index * len(bases)
            measured = {
                basis: all_probabilities[offset + basis_index]
                for basis_index, basis in enumerate(bases)
            }
            scores = scores_from_measurements(model, measured)
            prediction = int(np.argmax(scores))
            ordered = np.sort(scores)
            margin = float(ordered[-1] - ordered[-2])
            predictions.append(prediction)
            margins.append(margin)
            row: dict[str, Any] = {
                "point": index,
                "label": int(label),
                "prediction": prediction,
                "correct": int(prediction == label),
                "margin": margin,
            }
            for class_id, score in enumerate(scores):
                row[f"score_{class_id}"] = float(score)
            rows.append(row)
        summary = {
            "backend": f"origin-cloud:{self._backend_name}",
            "points": int(len(labels)),
            "shots_per_basis": shots,
            "measurement_bases": list(bases),
            "circuits_executed": len(programs),
            "accuracy": float(np.mean(np.asarray(predictions) == labels)),
            "mean_decision_margin": float(np.mean(margins)),
            "single_qubit_error": noise.single_qubit,
            "two_qubit_error": noise.two_qubit,
            "readout_error": noise.readout,
            "job_ids": job_ids,
        }
        return summary, rows, job_ids


__all__ = [
    "APPENDIX_SEED",
    "AppendixModel",
    "DepolarizingNoise",
    "OriginCloudBackend",
    "appendix_test_data",
    "author_u3",
    "bind_coordinates",
    "evaluate_local",
    "load_appendix_model",
    "measurement_probabilities",
    "sample_probabilities",
    "scores_from_measurements",
    "simulate_density_matrix",
    "stratified_subset",
]
