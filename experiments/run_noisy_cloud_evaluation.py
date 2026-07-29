#!/usr/bin/env python3
"""Evaluate a frozen appendix classifier under local or Origin Cloud noise."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from qs_project.noisy_cloud import (  # noqa: E402
    DepolarizingNoise,
    OriginCloudBackend,
    appendix_test_data,
    evaluate_local,
    load_appendix_model,
    stratified_subset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run finite-shot noisy evaluation for one frozen appendix result. "
            "The default reproduces the strongest crown result."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("local", "origin-cloud"),
        default="local",
    )
    parser.add_argument("--origin-backend", default="full_amplitude")
    parser.add_argument(
        "--list-origin-backends",
        action="store_true",
        help="authenticate, print currently available cloud backends, and exit",
    )
    parser.add_argument(
        "--chi",
        choices=("fidelity_chi", "weighted_fidelity_chi"),
        default="weighted_fidelity_chi",
    )
    parser.add_argument(
        "--problem",
        choices=("non convex", "crown", "sphere", "squares", "wavy lines"),
        default="crown",
    )
    parser.add_argument("--qubits", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument(
        "--entanglement",
        choices=("y", "n"),
        default="n",
    )
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="local only: use exact noisy probabilities instead of finite shots",
    )
    parser.add_argument(
        "--noise-levels",
        default="0,0.0005,0.001,0.002,0.005",
        help="comma-separated single-qubit depolarizing probabilities",
    )
    parser.add_argument("--two-qubit-multiplier", type=float, default=10.0)
    parser.add_argument("--readout-multiplier", type=float, default=2.0)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to results/noisy_cloud/<timestamp>",
    )
    return parser


def _noise_levels(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--noise-levels cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("--noise-levels contains duplicates")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("noise levels must lie in [0, 1]")
    return values


def _output_directory(requested: Path | None) -> Path:
    if requested is not None:
        output = requested.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "results" / "noisy_cloud" / stamp
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    x = [row["single_qubit_error"] for row in summaries]
    y = [row["accuracy"] for row in summaries]
    figure, axes = plt.subplots(figsize=(7.2, 4.4))
    axes.plot(x, y, "o-", color="#3567b7", linewidth=2)
    axes.set_xlabel("Single-qubit depolarizing probability")
    axes.set_ylabel("Classification accuracy")
    axes.set_ylim(0.0, 1.02)
    axes.grid(alpha=0.25)
    axes.set_title("Frozen quantum classifier under noise")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    if args.list_origin_backends:
        print(json.dumps(OriginCloudBackend.available_backends(), ensure_ascii=False, indent=2, default=str))
        return
    if args.backend == "origin-cloud" and args.exact:
        raise ValueError("--exact is only valid for the local backend")

    model = load_appendix_model(
        chi=args.chi,
        problem=args.problem,
        qubits=args.qubits,
        layers=args.layers,
        entanglement=args.entanglement,
    )
    all_x, all_y = appendix_test_data(args.problem)
    x, y, indices = stratified_subset(
        all_x,
        all_y,
        args.points,
        seed=args.selection_seed,
    )
    levels = _noise_levels(args.noise_levels)
    output = _output_directory(args.output_dir)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "test_set_indices": indices.tolist(),
                "selection_seed": args.selection_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cloud = (
        OriginCloudBackend(args.origin_backend)
        if args.backend == "origin-cloud"
        else None
    )
    summaries: list[dict[str, Any]] = []
    for level_index, level in enumerate(levels):
        readout = level * args.readout_multiplier
        if args.backend == "origin-cloud":
            readout = 0.0
        noise = DepolarizingNoise(
            single_qubit=level,
            two_qubit=min(1.0, level * args.two_qubit_multiplier),
            readout=min(1.0, readout),
        )
        jobs_path = output / "origin_job_ids.jsonl"

        def record_job(job_id: str, start: int, count: int) -> None:
            with jobs_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "noise_level": level,
                            "job_id": job_id,
                            "program_start": start,
                            "program_count": count,
                        }
                    )
                    + "\n"
                )

        if cloud is None:
            summary, rows = evaluate_local(
                model,
                x,
                y,
                noise=noise,
                shots=None if args.exact else args.shots,
                seed=args.sampling_seed + level_index,
            )
        else:
            summary, rows, _ = cloud.evaluate(
                model,
                x,
                y,
                shots=args.shots,
                noise=noise,
                batch_size=args.batch_size,
                on_job=record_job,
            )
        summary["recorded_noiseless_full_test_accuracy"] = model.recorded_accuracy
        summary["summary_path"] = str(model.summary_path)
        summaries.append(summary)
        _write_csv(output / f"points_noise_{level:.6f}.csv", rows)
        print(
            f"noise={level:.6g}, accuracy={summary['accuracy']:.4f}, "
            f"circuits={summary['circuits_executed']}"
        )

    _write_csv(output / "noise_sweep.csv", summaries)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "chi": model.chi,
            "problem": model.problem,
            "qubits": model.qubits,
            "layers": model.layers,
            "entanglement": model.entanglement,
            "classes": model.classes,
            "recorded_accuracy": model.recorded_accuracy,
            "summary_path": str(model.summary_path),
        },
        "evaluation": {
            "backend": args.backend,
            "origin_backend": args.origin_backend if cloud is not None else None,
            "points": args.points,
            "shots": None if args.exact else args.shots,
            "selection_seed": args.selection_seed,
            "sampling_seed": args.sampling_seed,
            "two_qubit_multiplier": args.two_qubit_multiplier,
            "readout_multiplier": (
                0.0 if cloud is not None else args.readout_multiplier
            ),
        },
        "noise_sweep": summaries,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(output / "noise_accuracy.png", summaries)
    print(f"results: {output}")


if __name__ == "__main__":
    main()
