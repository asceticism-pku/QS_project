#!/usr/bin/env python3
"""Final artifact verifier for the fixed 45-run project matrix.

This script is read-only with respect to ``results/raw``.  It writes a new,
timestamped project-audit directory and exits non-zero after writing the
evidence if any contract check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (str(ROOT), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from qs_project.core import (  # noqa: E402
    CONTROLLED_SEEDS,
    canonical_json_hash,
    git_revision,
    json_dump,
    make_circle_dataset,
    sha256_file,
)

PARITY_TOLERANCE = 1e-10
EXPECTED_STAGE_COUNTS = {"M1": 15, "M2": 15, "M3": 15}


@dataclass(frozen=True)
class ExpectedRun:
    experiment_id: str
    config_id: str
    seed: int
    qubits: int
    layers: int
    entanglement: str
    loss_id: str
    rng_mode: str
    run_kind: str = "optimizer"
    known_maxfun_nonconvergence: bool = False

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.experiment_id, self.config_id, self.seed


@dataclass(frozen=True)
class SingletonRun:
    spec: ExpectedRun
    result_path: Path
    result: dict[str, Any]
    config_path: Path
    config: dict[str, Any]


class InputRegistry:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def add(
        self,
        path: Path,
        *,
        role: str,
        code_revision: Any = None,
    ) -> None:
        if not path.is_file():
            return
        normalized = str(path.resolve())
        record = self._records.setdefault(
            normalized,
            {
                "path": normalized,
                "sha256": sha256_file(path),
                "roles": [],
                "code_revisions": [],
            },
        )
        if role not in record["roles"]:
            record["roles"].append(role)
        compact = compact_revision(code_revision)
        if compact and compact not in record["code_revisions"]:
            record["code_revisions"].append(compact)

    def records(self) -> list[dict[str, Any]]:
        records = []
        for record in self._records.values():
            copied = dict(record)
            copied["roles"] = sorted(copied["roles"])
            copied["code_revisions"] = sorted(
                copied["code_revisions"],
                key=lambda value: json.dumps(value, sort_keys=True),
            )
            records.append(copied)
        return sorted(records, key=lambda record: record["path"])


def expected_optimizer_runs() -> tuple[ExpectedRun, ...]:
    runs: list[ExpectedRun] = []
    for layers in (1, 2, 4, 8):
        runs.append(
            ExpectedRun(
                "M1",
                f"author-weighted-1q-l{layers}",
                30,
                1,
                layers,
                "n",
                "weighted_reduced_density",
                "legacy_exact",
                known_maxfun_nonconvergence=layers == 8,
            )
        )
    runs.append(
        ExpectedRun(
            "M1",
            "author-amplitude-1q-l4",
            30,
            1,
            4,
            "n",
            "legacy_amplitude",
            "legacy_exact",
        )
    )
    for loss_id in ("legacy_amplitude", "paper_squared"):
        for seed in CONTROLLED_SEEDS:
            runs.append(
                ExpectedRun(
                    "M1",
                    f"1q-l4-{loss_id}",
                    seed,
                    1,
                    4,
                    "n",
                    loss_id,
                    "controlled",
                )
            )

    m2_configs = (
        ("1q-l2-paper_squared", 1, "n"),
        ("2q-l2-separable-paper_squared", 2, "n"),
        ("2q-l2-cz-paper_squared", 2, "y"),
    )
    for config_id, qubits, entanglement in m2_configs:
        for seed in CONTROLLED_SEEDS:
            runs.append(
                ExpectedRun(
                    "M2",
                    config_id,
                    seed,
                    qubits,
                    2,
                    entanglement,
                    "paper_squared",
                    "controlled",
                )
            )

    for config_id in (
        "l4-to-l3-pruned",
        "l4-truncate-last",
        "l3-scratch",
    ):
        for seed in CONTROLLED_SEEDS:
            runs.append(
                ExpectedRun(
                    "M3",
                    config_id,
                    seed,
                    1,
                    3,
                    "n",
                    "paper_squared",
                    "controlled",
                )
            )
    if len(runs) != 45:
        raise AssertionError(f"internal optimizer matrix has {len(runs)} runs")
    return tuple(runs)


def compact_revision(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    keys = (
        "head",
        "branch",
        "dirty",
        "tracked_diff_sha256",
        "training_source_sha256",
        "compile_runner_sha256",
        "compile_module_sha256",
    )
    compact = {key: value[key] for key in keys if key in value}
    return compact or None


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def referenced_path(value: Any, *, repo_root: Path = ROOT) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def same_path(left: Any, right: Any, *, repo_root: Path = ROOT) -> bool:
    return referenced_path(left, repo_root=repo_root).resolve() == referenced_path(
        right, repo_root=repo_root
    ).resolve()


def finite_mapping(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if isinstance(item, dict):
            if not finite_mapping(item):
                return False
        elif (
            isinstance(item, bool)
            or not isinstance(item, numbers.Real)
            or not math.isfinite(float(item))
        ):
            return False
    return True


def validate_code_revision(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in (
        "head",
        "branch",
        "dirty",
        "tracked_diff_sha256",
        "training_source_sha256",
    ):
        if key not in value:
            return False
    return (
        isinstance(value["head"], str)
        and bool(value["head"])
        and isinstance(value["branch"], str)
        and isinstance(value["dirty"], bool)
        and isinstance(value["tracked_diff_sha256"], str)
        and isinstance(value["training_source_sha256"], str)
    )


def checked_artifact(
    payload: dict[str, Any],
    path_key: str,
    sha_key: str,
    *,
    label: str,
    issues: list[str],
    inputs: InputRegistry,
    code_revision: Any = None,
    repo_root: Path = ROOT,
) -> Path | None:
    if path_key not in payload or sha_key not in payload:
        issues.append(f"{label}: missing {path_key}/{sha_key}")
        return None
    path = referenced_path(payload[path_key], repo_root=repo_root)
    if not path.is_file():
        issues.append(f"{label}: missing artifact {path}")
        return None
    actual = sha256_file(path)
    expected = str(payload[sha_key])
    inputs.add(path, role=label, code_revision=code_revision)
    if actual != expected:
        issues.append(f"{label}: SHA-256 mismatch {actual} != {expected}")
    return path


def _config_fingerprint(config: dict[str, Any]) -> str:
    return canonical_json_hash(
        {
            key: value
            for key, value in config.items()
            if key
            not in {
                "command",
                "code_revision",
                "dataset_path",
                "config_fingerprint",
            }
        }
    )


def _validate_optimizer_run(
    *,
    spec: ExpectedRun | None,
    result_path: Path,
    result: dict[str, Any],
    config_path: Path | None,
    config: dict[str, Any] | None,
    dataset_hash: str,
    inputs: InputRegistry,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str], bool]:
    issues: list[str] = []
    identity = (
        str(result.get("experiment_id", "")),
        str(result.get("config_id", "")),
        result.get("seed"),
    )
    label = "/".join(map(str, identity))
    revision = result.get("code_revision")
    inputs.add(result_path, role=f"optimizer-result:{label}", code_revision=revision)
    if spec is None:
        issues.append("unexpected optimizer identity")
    else:
        expected_result_fields = {
            "experiment_id": spec.experiment_id,
            "config_id": spec.config_id,
            "seed": spec.seed,
            "loss_id": spec.loss_id,
            "rng_mode": spec.rng_mode,
        }
        for key, expected in expected_result_fields.items():
            if result.get(key) != expected:
                issues.append(
                    f"result {key}={result.get(key)!r}, expected {expected!r}"
                )

    if result.get("dataset_hash") != dataset_hash:
        issues.append("result dataset hash mismatch")
    if result.get("verification") != "artifacts-verified":
        issues.append(f"result verification={result.get('verification')!r}")
    if not validate_code_revision(revision):
        issues.append("missing or malformed result code revision")
    command = result.get("command")
    if not isinstance(command, str) or not command.strip():
        issues.append("missing result command")

    if config_path is None or config is None:
        issues.append("missing config.json")
    else:
        inputs.add(
            config_path,
            role=f"optimizer-config:{label}",
            code_revision=config.get("code_revision"),
        )
        if spec is not None:
            expected_config_fields = {
                "experiment_id": spec.experiment_id,
                "config_id": spec.config_id,
                "init_seed": spec.seed,
                "qubits": spec.qubits,
                "layers": spec.layers,
                "entanglement": spec.entanglement,
                "loss_id": spec.loss_id,
                "rng_mode": spec.rng_mode,
                "run_kind": spec.run_kind,
            }
            for key, expected in expected_config_fields.items():
                if config.get(key) != expected:
                    issues.append(
                        f"config {key}={config.get(key)!r}, expected {expected!r}"
                    )
        if config.get("dataset_hash") != dataset_hash:
            issues.append("config dataset hash mismatch")
        if config.get("command") != command:
            issues.append("config/result command mismatch")
        if config.get("code_revision") != revision:
            issues.append("config/result code revision mismatch")
        fingerprint = _config_fingerprint(config)
        if config.get("config_fingerprint") != fingerprint:
            issues.append("config fingerprint mismatch")
        if result.get("config_fingerprint") != fingerprint:
            issues.append("result config fingerprint mismatch")

    command_path = result_path.parent / "command.txt"
    if not command_path.is_file():
        issues.append("missing command.txt")
    else:
        inputs.add(command_path, role=f"optimizer-command:{label}", code_revision=revision)
        if command_path.read_text(encoding="utf-8").strip() != str(command):
            issues.append("command.txt/result command mismatch")

    checkpoint_path = checked_artifact(
        result,
        "checkpoint",
        "checkpoint_sha256",
        label=f"optimizer-checkpoint:{label}",
        issues=issues,
        inputs=inputs,
        code_revision=revision,
        repo_root=repo_root,
    )
    initial_checkpoint_path = checked_artifact(
        result,
        "initial_checkpoint",
        "initial_checkpoint_sha256",
        label=f"optimizer-initial-checkpoint:{label}",
        issues=issues,
        inputs=inputs,
        code_revision=revision,
        repo_root=repo_root,
    )
    if checkpoint_path is not None and checkpoint_path.parent != result_path.parent:
        issues.append("checkpoint is not stored beside its result")
    if (
        initial_checkpoint_path is not None
        and initial_checkpoint_path.parent != result_path.parent
    ):
        issues.append("initial checkpoint is not stored beside its result")

    checked_artifact(
        result,
        "dataset_artifact",
        "dataset_artifact_sha256",
        label=f"optimizer-dataset:{label}",
        issues=issues,
        inputs=inputs,
        code_revision=revision,
        repo_root=repo_root,
    )

    for metrics_key in ("train_metrics", "test_metrics"):
        metrics = result.get(metrics_key)
        if not finite_mapping(metrics):
            issues.append(f"{metrics_key} is missing or non-finite")
        elif not 0.0 <= float(metrics.get("accuracy", math.nan)) <= 1.0:
            issues.append(f"{metrics_key}.accuracy is outside [0,1]")

    optimizer = result.get("optimizer")
    known_nonconvergence = False
    if not isinstance(optimizer, dict):
        issues.append("missing optimizer payload")
        optimizer = {}
    else:
        for key in ("status", "nfev", "fun", "initial_fun", "nit"):
            value = optimizer.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
            ):
                issues.append(f"optimizer.{key} is missing or non-finite")
        if not isinstance(optimizer.get("nfev"), int) or optimizer.get("nfev", 0) <= 0:
            issues.append("optimizer.nfev must be a positive integer")
        if not isinstance(optimizer.get("status"), int):
            issues.append("optimizer.status must be an integer")
        if not isinstance(optimizer.get("message"), str) or not optimizer.get(
            "message", ""
        ).strip():
            issues.append("optimizer.message is missing")

        if spec is not None and spec.known_maxfun_nonconvergence:
            maxfun = config.get("maxfun") if isinstance(config, dict) else None
            known_nonconvergence = (
                optimizer.get("success") is False
                and optimizer.get("status") == 1
                and isinstance(optimizer.get("nfev"), int)
                and isinstance(maxfun, int)
                and optimizer["nfev"] >= maxfun
                and "LIMIT" in str(optimizer.get("message", "")).upper()
            )
            if not known_nonconvergence:
                issues.append("known L8 maxfun non-convergence signature changed")
        elif optimizer.get("success") is not True or optimizer.get("status") != 0:
            issues.append(
                "optimizer did not converge successfully outside the L8 exception"
            )

    row = {
        "experiment_id": result.get("experiment_id"),
        "config_id": result.get("config_id"),
        "seed": result.get("seed"),
        "expected_identity": spec is not None,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": (
            sha256_file(config_path) if config_path and config_path.is_file() else None
        ),
        "run_kind": config.get("run_kind") if isinstance(config, dict) else None,
        "qubits": config.get("qubits") if isinstance(config, dict) else None,
        "layers": config.get("layers") if isinstance(config, dict) else None,
        "entanglement": (
            config.get("entanglement") if isinstance(config, dict) else None
        ),
        "loss_id": result.get("loss_id"),
        "rng_mode": result.get("rng_mode"),
        "dataset_hash": result.get("dataset_hash"),
        "verification": result.get("verification"),
        "optimizer_success": optimizer.get("success"),
        "optimizer_status": optimizer.get("status"),
        "optimizer_message": optimizer.get("message"),
        "nfev": optimizer.get("nfev"),
        "nit": optimizer.get("nit"),
        "final_loss": optimizer.get("fun"),
        "train_accuracy": (
            result.get("train_metrics", {}).get("accuracy")
            if isinstance(result.get("train_metrics"), dict)
            else None
        ),
        "test_accuracy": (
            result.get("test_metrics", {}).get("accuracy")
            if isinstance(result.get("test_metrics"), dict)
            else None
        ),
        "checkpoint": result.get("checkpoint"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "initial_checkpoint": result.get("initial_checkpoint"),
        "initial_checkpoint_sha256": result.get("initial_checkpoint_sha256"),
        "command": command,
        "revision_head": (
            revision.get("head") if isinstance(revision, dict) else None
        ),
        "revision_branch": (
            revision.get("branch") if isinstance(revision, dict) else None
        ),
        "training_source_sha256": (
            revision.get("training_source_sha256")
            if isinstance(revision, dict)
            else None
        ),
        "known_maxfun_nonconvergence": known_nonconvergence,
        "audit_passed": not issues,
        "audit_issues": " | ".join(issues),
    }
    return row, issues, known_nonconvergence


def audit_optimizer_matrix(
    *,
    raw_root: Path,
    dataset_hash: str,
    inputs: InputRegistry,
    repo_root: Path = ROOT,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, str, int], SingletonRun],
]:
    specs = expected_optimizer_runs()
    expected = {spec.identity: spec for spec in specs}
    groups: dict[
        tuple[str, str, int],
        list[tuple[Path, dict[str, Any], Path | None, dict[str, Any] | None]],
    ] = {}
    scan_issues: list[str] = []

    for stage in ("M1", "M2", "M3"):
        stage_root = raw_root / stage
        for result_path in sorted(stage_root.glob("**/result.json")):
            try:
                result = read_json_object(result_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                scan_issues.append(f"cannot read {result_path}: {exc}")
                continue
            config_path = result_path.parent / "config.json"
            config: dict[str, Any] | None = None
            if config_path.is_file():
                try:
                    config = read_json_object(config_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    scan_issues.append(f"cannot read {config_path}: {exc}")
            is_optimizer = isinstance(result.get("optimizer"), dict) or (
                isinstance(config, dict) and config.get("run_kind") == "optimizer"
            )
            if not is_optimizer:
                continue
            try:
                identity = (
                    str(result["experiment_id"]),
                    str(result["config_id"]),
                    int(result["seed"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                scan_issues.append(
                    f"optimizer result has invalid identity {result_path}: {exc}"
                )
                continue
            groups.setdefault(identity, []).append(
                (result_path, result, config_path if config_path.is_file() else None, config)
            )

    rows: list[dict[str, Any]] = []
    issues = list(scan_issues)
    known_nonconverged: list[dict[str, Any]] = []
    singleton_runs: dict[tuple[str, str, int], SingletonRun] = {}
    all_identities = sorted(set(expected) | set(groups))
    for identity in all_identities:
        candidates = groups.get(identity, [])
        spec = expected.get(identity)
        if spec is not None and len(candidates) == 0:
            issues.append(f"missing optimizer run {identity}")
        if len(candidates) > 1:
            issues.append(
                f"duplicate optimizer run {identity}: "
                f"{[str(candidate[0]) for candidate in candidates]}"
            )
        if spec is None:
            issues.append(f"unexpected optimizer run {identity}")
        for result_path, result, config_path, config in candidates:
            row, row_issues, known = _validate_optimizer_run(
                spec=spec,
                result_path=result_path,
                result=result,
                config_path=config_path,
                config=config,
                dataset_hash=dataset_hash,
                inputs=inputs,
                repo_root=repo_root,
            )
            if len(candidates) > 1:
                row_issues.append("duplicate identity")
                row["audit_passed"] = False
                row["audit_issues"] = " | ".join(row_issues)
            rows.append(row)
            issues.extend(f"{identity}: {issue}" for issue in row_issues)
            if known:
                known_nonconverged.append(
                    {
                        "identity": list(identity),
                        "status": result["optimizer"]["status"],
                        "nfev": result["optimizer"]["nfev"],
                        "maxfun": config["maxfun"] if config else None,
                        "message": result["optimizer"]["message"],
                        "result": str(result_path),
                    }
                )
        if spec is not None and len(candidates) == 1:
            result_path, result, config_path, config = candidates[0]
            if config_path is not None and config is not None:
                singleton_runs[identity] = SingletonRun(
                    spec,
                    result_path,
                    result,
                    config_path,
                    config,
                )

    actual_stage_counts = {
        stage: sum(1 for identity, values in groups.items() if identity[0] == stage for _ in values)
        for stage in EXPECTED_STAGE_COUNTS
    }
    actual_count = sum(len(values) for values in groups.values())
    exact_identities = (
        set(groups) == set(expected)
        and all(len(values) == 1 for values in groups.values())
    )
    known_l8_exactly_once = (
        len(known_nonconverged) == 1
        and known_nonconverged[0]["identity"]
        == ["M1", "author-weighted-1q-l8", 30]
    )
    passed = (
        not issues
        and actual_count == 45
        and actual_stage_counts == EXPECTED_STAGE_COUNTS
        and exact_identities
        and known_l8_exactly_once
    )
    report = {
        "passed": passed,
        "expected_count": 45,
        "actual_count": actual_count,
        "expected_stage_counts": EXPECTED_STAGE_COUNTS,
        "actual_stage_counts": actual_stage_counts,
        "exact_identity_once": exact_identities,
        "known_l8_nonconvergence_preserved": known_l8_exactly_once,
        "known_nonconverged_runs": known_nonconverged,
        "issues": issues,
    }
    rows.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            str(row["config_id"]),
            int(row["seed"]) if row["seed"] is not None else -1,
            str(row["result_path"]),
        )
    )
    return report, rows, singleton_runs


def _latest_result_candidates(
    raw_root: Path,
    experiment_id: str,
    config_id: str,
) -> list[Path]:
    root = raw_root / experiment_id / config_id
    return sorted(
        root.glob("**/result.json"),
        key=lambda path: (path.parent.name, str(path)),
    )


def _expected_compile_sources(
    stage: str,
) -> dict[tuple[str, int], tuple[str, str, int]]:
    if stage == "M2":
        model_sources = (
            ("1q-l4-paper_squared", "M1", "1q-l4-paper_squared"),
            ("1q-l2-paper_squared", "M2", "1q-l2-paper_squared"),
            (
                "2q-l2-separable-paper_squared",
                "M2",
                "2q-l2-separable-paper_squared",
            ),
            ("2q-l2-cz-paper_squared", "M2", "2q-l2-cz-paper_squared"),
        )
    elif stage == "M3":
        model_sources = (
            ("l4-base", "M1", "1q-l4-paper_squared"),
            ("l4-to-l3-pruned", "M3", "l4-to-l3-pruned"),
            ("l4-truncate-last", "M3", "l4-truncate-last"),
            ("l3-scratch", "M3", "l3-scratch"),
        )
    else:
        raise ValueError(f"unsupported compile stage: {stage}")
    return {
        (model_id, seed): (source_stage, source_config, seed)
        for model_id, source_stage, source_config in model_sources
        for seed in CONTROLLED_SEEDS
    }


def audit_compile_result(
    *,
    stage: str,
    raw_root: Path,
    dataset_hash: str,
    optimizer_runs: dict[tuple[str, str, int], SingletonRun],
    inputs: InputRegistry,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    candidates = _latest_result_candidates(raw_root, stage, "compile-audit")
    report: dict[str, Any] = {
        "passed": False,
        "candidate_count": len(candidates),
        "candidates": [],
        "selected_result": None,
        "issues": [],
    }
    for path in candidates:
        try:
            payload = read_json_object(path)
            status = payload.get("verification")
        except Exception as exc:  # preserved in the audit report
            status = f"unreadable: {exc}"
        report["candidates"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "verification": status,
            }
        )
    if not candidates:
        report["issues"].append(f"missing {stage} compile-audit result")
        return report

    result_path = candidates[-1]
    result = read_json_object(result_path)
    revision = result.get("code_revision")
    inputs.add(
        result_path,
        role=f"{stage}-compile-result",
        code_revision=revision,
    )
    report["selected_result"] = str(result_path)
    report["selected_result_sha256"] = sha256_file(result_path)
    issues: list[str] = []
    scientific_findings: list[dict[str, Any]] = []
    if result.get("experiment_id") != stage or result.get("config_id") != "compile-audit":
        issues.append("compile result identity mismatch")
    if result.get("dataset_hash") != dataset_hash:
        issues.append("compile dataset hash mismatch")
    if result.get("checkpoint_count") != 20:
        issues.append("compile checkpoint_count must be 20")
    if result.get("point_count_per_checkpoint") != 100:
        issues.append("compile point_count_per_checkpoint must be 100")
    if result.get("row_count") != 4000:
        issues.append("compile row_count must be 4000")
    if result.get("all_label_parity") is not True:
        issues.append("compile label parity failed")
    max_error = result.get("max_probability_error")
    max_error_is_finite = not (
        isinstance(max_error, bool)
        or not isinstance(max_error, numbers.Real)
        or not math.isfinite(float(max_error))
    )
    if not max_error_is_finite:
        issues.append("compile max probability error is not finite")

    if stage == "M2":
        if result.get("verification") != "parity-failed":
            issues.append(
                "M2 compile verification must preserve the parity-failed result"
            )
        if result.get("all_probability_parity") is not False:
            issues.append("M2 compile all_probability_parity must be false")
        if max_error_is_finite and float(max_error) < PARITY_TOLERANCE:
            issues.append(
                "M2 compile max probability error does not explain parity-failed"
            )
        scientific_findings.append(
            {
                "finding": "probability-parity-failed",
                "classification": "preserved-scientific-negative-result",
                "expected_for_stage": True,
                "affects_artifact_integrity": False,
                "observed": (
                    result.get("verification") == "parity-failed"
                    and result.get("all_probability_parity") is False
                    and max_error_is_finite
                    and float(max_error) >= PARITY_TOLERANCE
                ),
                "tolerance": PARITY_TOLERANCE,
                "max_probability_error": max_error,
            }
        )
    else:
        if result.get("verification") != "artifacts-verified":
            issues.append(f"compile verification={result.get('verification')!r}")
        if result.get("all_probability_parity") is not True:
            issues.append("compile probability parity failed")
        if max_error_is_finite and float(max_error) >= PARITY_TOLERANCE:
            issues.append("compile max probability error is not below 1e-10")
    command = result.get("command")
    if not isinstance(command, str) or not command.strip():
        issues.append("compile command is missing")
    if not isinstance(revision, dict) or not revision.get("head"):
        issues.append("compile code revision is missing")

    checked_paths: dict[str, Path | None] = {}
    for path_key, hash_key in (
        ("rows", "rows_sha256"),
        ("summary", "summary_sha256"),
        ("checkpoint_records", "checkpoint_records_sha256"),
        ("compile_indices", "compile_indices_file_sha256"),
    ):
        checked_paths[path_key] = checked_artifact(
            result,
            path_key,
            hash_key,
            label=f"{stage}-compile-{path_key}",
            issues=issues,
            inputs=inputs,
            code_revision=revision,
            repo_root=repo_root,
        )

    actual_row_count: int | None = None
    rows_path = checked_paths["rows"]
    if rows_path is not None and rows_path.is_file():
        actual_row_count = 0
        try:
            with rows_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        issues.append(
                            f"compile rows contains blank line {line_number}"
                        )
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        issues.append(
                            f"compile row {line_number} is not a JSON object"
                        )
                    actual_row_count += 1
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(f"compile rows is unreadable: {exc}")
        if actual_row_count != 4000:
            issues.append(
                f"compile rows artifact has {actual_row_count} rows, expected 4000"
            )

    records_path = checked_paths["checkpoint_records"]
    records: list[Any] = []
    if records_path is not None and records_path.is_file():
        loaded = json.loads(records_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = loaded
        else:
            issues.append("compile checkpoint_records is not a list")
    record_map: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(f"compile checkpoint record {index} is not an object")
            continue
        model_id = str(record.get("model_id", record.get("config_id", "")))
        try:
            key = model_id, int(record["seed"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"compile checkpoint record {index} has invalid identity")
            continue
        if key in record_map:
            issues.append(f"duplicate compile checkpoint record {key}")
        record_map[key] = record

    expected_sources = _expected_compile_sources(stage)
    if set(record_map) != set(expected_sources):
        issues.append(
            "compile checkpoint identities differ from the expected 20-record matrix"
        )
    for key, source_identity in expected_sources.items():
        record = record_map.get(key)
        source = optimizer_runs.get(source_identity)
        if record is None or source is None:
            issues.append(f"cannot verify compile checkpoint reuse for {key}")
            continue
        if not same_path(
            record.get("source_result", ""),
            source.result_path,
            repo_root=repo_root,
        ):
            issues.append(f"compile source result mismatch for {key}")
        if not same_path(
            record.get("checkpoint", ""),
            source.result.get("checkpoint", ""),
            repo_root=repo_root,
        ):
            issues.append(f"compile checkpoint path mismatch for {key}")
        if record.get("checkpoint_sha256") != source.result.get(
            "checkpoint_sha256"
        ):
            issues.append(f"compile checkpoint hash mismatch for {key}")

    report.update(
        {
            "verification": result.get("verification"),
            "checkpoint_count": result.get("checkpoint_count"),
            "point_count_per_checkpoint": result.get(
                "point_count_per_checkpoint"
            ),
            "row_count": result.get("row_count"),
            "actual_row_count": actual_row_count,
            "all_probability_parity": result.get("all_probability_parity"),
            "all_label_parity": result.get("all_label_parity"),
            "max_probability_error": max_error,
            "scientific_findings": scientific_findings,
            "checkpoint_reuse_verified": (
                set(record_map) == set(expected_sources)
                and not any("compile checkpoint" in issue for issue in issues)
            ),
            "issues": issues,
            "passed": not issues,
        }
    )
    return report


def audit_m4_result(
    *,
    raw_root: Path,
    dataset_hash: str,
    optimizer_runs: dict[tuple[str, str, int], SingletonRun],
    inputs: InputRegistry,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    candidates = _latest_result_candidates(raw_root, "M4", "fixed-adaptive-shots")
    report: dict[str, Any] = {
        "passed": False,
        "candidate_count": len(candidates),
        "candidates": [],
        "selected_result": None,
        "issues": [],
    }
    for path in candidates:
        try:
            payload = read_json_object(path)
            status = payload.get("verification")
        except Exception as exc:
            status = f"unreadable: {exc}"
        report["candidates"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "verification": status,
            }
        )
    if not candidates:
        report["issues"].append("missing M4 fixed-adaptive-shots result")
        return report

    result_path = candidates[-1]
    result = read_json_object(result_path)
    revision = result.get("code_revision")
    inputs.add(result_path, role="M4-result", code_revision=revision)
    issues: list[str] = []
    report["selected_result"] = str(result_path)
    report["selected_result_sha256"] = sha256_file(result_path)
    if result.get("experiment_id") != "M4" or result.get("config_id") != (
        "fixed-adaptive-shots"
    ):
        issues.append("M4 result identity mismatch")
    if result.get("dataset_hash") != dataset_hash:
        issues.append("M4 dataset hash mismatch")
    if result.get("verification") != "artifacts-verified":
        issues.append(f"M4 verification={result.get('verification')!r}")
    if result.get("evidence_label") != "shot-simulation":
        issues.append("M4 evidence label must be shot-simulation")
    if result.get("optimizer_runs") != 0:
        issues.append("M4 optimizer_runs must be zero")
    if result.get("nfev") != 0:
        issues.append("M4 nfev must be zero")
    if result.get("checkpoint_count") != 10:
        issues.append("M4 checkpoint_count must be 10")
    if result.get("campaign_repeats_per_checkpoint") != 100:
        issues.append("M4 repeats per checkpoint must be 100")
    if result.get("campaign_count") != 1000:
        issues.append("M4 campaign_count must be 1000")
    if result.get("campaign_metric_row_count") != 4000:
        issues.append("M4 campaign metric row count must be 4000")
    if result.get("evaluation_source_stable") is not True:
        issues.append("M4 evaluation source changed during the run")
    if not isinstance(result.get("command"), str) or not result["command"].strip():
        issues.append("M4 command is missing")
    if not isinstance(revision, dict) or not revision.get("head"):
        issues.append("M4 code revision is missing")

    artifact_pairs = (
        ("campaign_metrics_csv", "campaign_metrics_csv_sha256"),
        ("campaign_metrics_json", "campaign_metrics_json_sha256"),
        ("campaign_metrics_jsonl", "campaign_metrics_jsonl_sha256"),
        ("checkpoint_records", "checkpoint_records_sha256"),
        ("config", "config_sha256"),
        ("dataset_artifact", "dataset_artifact_sha256"),
        ("environment", "environment_sha256"),
        ("eval_indices", "eval_indices_file_sha256"),
        ("exact_probabilities", "exact_probabilities_sha256"),
        ("metrics", "metrics_sha256"),
    )
    checked_paths: dict[str, Path] = {}
    for path_key, hash_key in artifact_pairs:
        path = checked_artifact(
            result,
            path_key,
            hash_key,
            label=f"M4-{path_key}",
            issues=issues,
            inputs=inputs,
            code_revision=revision,
            repo_root=repo_root,
        )
        if path is not None:
            checked_paths[path_key] = path

    config_path = checked_paths.get("config")
    if config_path is not None:
        config = read_json_object(config_path)
        if config.get("optimizer_runs") != 0:
            issues.append("M4 config optimizer_runs must be zero")
        if config.get("run_kind") != "frozen-checkpoint-shot-evaluation":
            issues.append("M4 config run_kind mismatch")
        if config.get("dataset_hash") != dataset_hash:
            issues.append("M4 config dataset hash mismatch")
        if config.get("command") != result.get("command"):
            issues.append("M4 config/result command mismatch")

    command_path = result_path.parent / "command.txt"
    if not command_path.is_file():
        issues.append("M4 command.txt is missing")
    else:
        inputs.add(command_path, role="M4-command", code_revision=revision)
        if command_path.read_text(encoding="utf-8").strip() != str(
            result.get("command")
        ):
            issues.append("M4 command.txt/result command mismatch")

    records: list[Any] = []
    records_path = checked_paths.get("checkpoint_records")
    if records_path is not None:
        loaded = json.loads(records_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = loaded
        else:
            issues.append("M4 checkpoint_records is not a list")

    expected_sources = {
        ("l4-base", seed): ("M1", "1q-l4-paper_squared", seed)
        for seed in CONTROLLED_SEEDS
    }
    expected_sources.update(
        {
            ("l4-to-l3-pruned", seed): ("M3", "l4-to-l3-pruned", seed)
            for seed in CONTROLLED_SEEDS
        }
    )
    record_map: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(f"M4 checkpoint record {index} is not an object")
            continue
        try:
            key = str(record["model_id"]), int(record["training_seed"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"M4 checkpoint record {index} has invalid identity")
            continue
        if key in record_map:
            issues.append(f"duplicate M4 checkpoint record {key}")
        record_map[key] = record
    if set(record_map) != set(expected_sources):
        issues.append("M4 checkpoint identities differ from the expected ten")

    for key, source_identity in expected_sources.items():
        record = record_map.get(key)
        source = optimizer_runs.get(source_identity)
        if record is None or source is None:
            issues.append(f"cannot verify M4 checkpoint reuse for {key}")
            continue
        expected_stage, expected_config, expected_seed = source_identity
        expected_checkpoint = source.result.get("checkpoint")
        expected_hash = source.result.get("checkpoint_sha256")
        expected_result_hash = sha256_file(source.result_path)
        field_expectations = {
            "source_experiment_id": expected_stage,
            "source_config_id": expected_config,
            "training_seed": expected_seed,
            "checkpoint_sha256": expected_hash,
            "source_result_sha256": expected_result_hash,
            "dataset_hash": dataset_hash,
            "verified_candidate_count": 1,
        }
        for field, expected in field_expectations.items():
            if record.get(field) != expected:
                issues.append(f"M4 {key} {field} mismatch")
        for field in ("source_result", "selected_result"):
            if not same_path(
                record.get(field, ""),
                source.result_path,
                repo_root=repo_root,
            ):
                issues.append(f"M4 {key} {field} does not reuse source result")
        for field in ("checkpoint", "checkpoint_recorded_path"):
            if not same_path(
                record.get(field, ""),
                expected_checkpoint,
                repo_root=repo_root,
            ):
                issues.append(f"M4 {key} {field} does not reuse source checkpoint")

    checkpoint_reuse_verified = (
        set(record_map) == set(expected_sources)
        and not any("M4 checkpoint" in issue or "M4 (" in issue for issue in issues)
    )
    report.update(
        {
            "verification": result.get("verification"),
            "optimizer_runs": result.get("optimizer_runs"),
            "nfev": result.get("nfev"),
            "checkpoint_count": result.get("checkpoint_count"),
            "checkpoint_reuse_verified": checkpoint_reuse_verified,
            "issues": issues,
            "passed": not issues,
        }
    )
    return report


def unique_output_directory(summary_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = summary_root / "project" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("optimizer_runs.csv cannot be empty")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_final_audit(
    *,
    raw_root: Path | None = None,
    summary_root: Path | None = None,
    expected_dataset_hash: str | None = None,
    repo_root: Path = ROOT,
    fail_on_error: bool = True,
) -> Path:
    raw_root = raw_root or repo_root / "results" / "raw"
    summary_root = summary_root or repo_root / "results" / "summary"
    dataset_hash = expected_dataset_hash or make_circle_dataset().dataset_hash
    inputs = InputRegistry()
    optimizer_report, optimizer_rows, singleton_runs = audit_optimizer_matrix(
        raw_root=raw_root,
        dataset_hash=dataset_hash,
        inputs=inputs,
        repo_root=repo_root,
    )
    m2_compile = audit_compile_result(
        stage="M2",
        raw_root=raw_root,
        dataset_hash=dataset_hash,
        optimizer_runs=singleton_runs,
        inputs=inputs,
        repo_root=repo_root,
    )
    m3_compile = audit_compile_result(
        stage="M3",
        raw_root=raw_root,
        dataset_hash=dataset_hash,
        optimizer_runs=singleton_runs,
        inputs=inputs,
        repo_root=repo_root,
    )
    m4_report = audit_m4_result(
        raw_root=raw_root,
        dataset_hash=dataset_hash,
        optimizer_runs=singleton_runs,
        inputs=inputs,
        repo_root=repo_root,
    )

    checks = {
        "optimizer_matrix_exactly_45": optimizer_report["passed"],
        "known_l8_maxfun_nonconvergence_preserved": optimizer_report[
            "known_l8_nonconvergence_preserved"
        ],
        "M2_compile_artifacts_verified": m2_compile["passed"],
        "M3_compile_artifacts_verified": m3_compile["passed"],
        "M4_frozen_checkpoint_evaluation_verified": m4_report["passed"],
    }
    scientific_findings = {
        "M2_compile": m2_compile.get("scientific_findings", []),
        "M3_compile": m3_compile.get("scientific_findings", []),
    }
    all_issues = [
        *optimizer_report["issues"],
        *(f"M2 compile: {issue}" for issue in m2_compile["issues"]),
        *(f"M3 compile: {issue}" for issue in m3_compile["issues"]),
        *(f"M4: {issue}" for issue in m4_report["issues"]),
    ]
    passed = all(checks.values()) and not all_issues
    output = unique_output_directory(summary_root)
    optimizer_csv = output / "optimizer_runs.csv"
    write_csv(optimizer_csv, optimizer_rows)

    revision = git_revision() if repo_root == ROOT else {}
    compact_current_revision = compact_revision(revision) or {}
    compact_current_revision["verifier_sha256"] = sha256_file(
        Path(__file__).resolve()
    )
    audit_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": "artifacts-verified" if passed else "failed-audit",
        "passed": passed,
        "command": " ".join([sys.executable, *sys.argv]),
        "dataset_hash": dataset_hash,
        "matrix_contract": {
            "expected_optimizer_runs": 45,
            "expected_by_stage": EXPECTED_STAGE_COUNTS,
            "excluded_from_optimizer_count": [
                "P0",
                "M1/loss-unit-audit",
                "M3/pruning-selection",
                "M2/compile-audit",
                "M3/compile-audit",
                "M4",
            ],
        },
        "checks": checks,
        "optimizer_audit": optimizer_report,
        "compile_audits": {"M2": m2_compile, "M3": m3_compile},
        "scientific_findings": scientific_findings,
        "M4_audit": m4_report,
        "issues": all_issues,
        "input_count": len(inputs.records()),
        "inputs": inputs.records(),
        "code_revision": compact_current_revision,
        "outputs": {
            "optimizer_runs_csv": str(optimizer_csv),
            "optimizer_runs_csv_sha256": sha256_file(optimizer_csv),
            "final_audit": str(output / "final_audit.json"),
            "sha256_manifest": str(output / "SHA256SUMS"),
        },
    }
    final_audit_path = output / "final_audit.json"
    json_dump(final_audit_path, audit_payload)
    manifest_entries = {
        "final_audit.json": sha256_file(final_audit_path),
        "optimizer_runs.csv": sha256_file(optimizer_csv),
    }
    manifest_path = output / "SHA256SUMS"
    with manifest_path.open("x", encoding="utf-8") as handle:
        for filename in sorted(manifest_entries):
            handle.write(f"{manifest_entries[filename]}  {filename}\n")

    print(
        json.dumps(
            {
                "final_audit": str(final_audit_path),
                "sha256_manifest": str(manifest_path),
                "passed": passed,
                "failed_checks": [
                    name for name, check_passed in checks.items() if not check_passed
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if fail_on_error and not passed:
        raise SystemExit(
            "project artifact audit failed; timestamped audit outputs were preserved"
        )
    return final_audit_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="write the audit but return zero even when checks fail",
    )
    args = parser.parse_args()
    run_final_audit(fail_on_error=not args.no_fail)


if __name__ == "__main__":
    main()
