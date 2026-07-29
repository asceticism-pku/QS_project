from __future__ import annotations

import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "doc" / "大作业报告"
FIGURE_DIR = DOC_DIR / "figures" / "appendix_results"
TEX_PATH = DOC_DIR / "appendix_experiment_results.tex"
MAIN_TEX_PATH = DOC_DIR / "大作业报告.tex"
CSV_PATH = DOC_DIR / "appendix_experiment_results.csv"
STATISTICS_CSV_PATH = DOC_DIR / "appendix_experiment_statistics.csv"
MAIN_TEX_BEGIN_MARKER = "% BEGIN AUTO-GENERATED APPENDIX EXPERIMENT RESULTS"
MAIN_TEX_END_MARKER = "% END AUTO-GENERATED APPENDIX EXPERIMENT RESULTS"

SEED = 30
LAYERS = (1, 2, 3, 4, 5, 6, 8, 10)
PROBLEMS = {
    "non convex": ("非凸边界问题", "non-convex", "non_convex"),
    "crown": ("二分类圆环问题", "binary-annulus", "crown"),
    "sphere": ("三维球问题", "sphere", "sphere"),
    "squares": ("四分类方块问题", "squares", "squares"),
    "wavy lines": ("四分类波浪线问题", "wavy-lines", "wavy_lines"),
}
COLUMNS = (
    ("fidelity_chi", 1, "not_entangled"),
    ("fidelity_chi", 2, "not_entangled"),
    ("fidelity_chi", 2, "entangled"),
    ("weighted_fidelity_chi", 1, "not_entangled"),
    ("weighted_fidelity_chi", 2, "not_entangled"),
    ("weighted_fidelity_chi", 2, "entangled"),
    ("weighted_fidelity_chi", 4, "not_entangled"),
    ("weighted_fidelity_chi", 4, "entangled"),
)
COMMON_COLUMNS = COLUMNS[:6]


@dataclass(frozen=True)
class Result:
    chi: str
    problem: str
    qubits: int
    layers: int
    entanglement: str
    accuracy: float
    summary_path: Path
    image_path: Path

    @property
    def key(self) -> tuple[str, str, int, int, str]:
        return (
            self.chi,
            self.problem,
            self.qubits,
            self.layers,
            self.entanglement,
        )


def extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing field matching {pattern!r}")
    return match.group(1).strip()


def read_results() -> list[Result]:
    results: list[Result] = []
    filename = f"appendix_seed_{SEED}_summary.txt"
    for chi in ("fidelity_chi", "weighted_fidelity_chi"):
        for summary_path in sorted((ROOT / chi).rglob(filename)):
            text = summary_path.read_text(encoding="utf-8")
            problem = extract(r"^Problem = (.+)$", text)
            qubits = int(extract(r"^Number of qubits = (\d+)$", text))
            layers = int(extract(r"^Number of layers = (\d+)$", text))
            accuracy = float(extract(r"^acc_test = ([0-9.eE+-]+)$", text))
            entanglement = (
                "entangled"
                if qubits > 1 and "entangled" in summary_path.parts
                and "not_entangled" not in summary_path.parts
                else "not_entangled"
            )
            image_path = summary_path.with_name(
                summary_path.name.replace("_summary.txt", ".png")
            )
            results.append(
                Result(
                    chi=chi,
                    problem=problem,
                    qubits=qubits,
                    layers=layers,
                    entanglement=entanglement,
                    accuracy=accuracy,
                    summary_path=summary_path,
                    image_path=image_path,
                )
            )
    return results


def validate(results: list[Result]) -> dict[tuple[str, str, int, int, str], Result]:
    lookup: dict[tuple[str, str, int, int, str], Result] = {}
    for result in results:
        if result.key in lookup:
            raise ValueError(f"Duplicate result for {result.key}")
        lookup[result.key] = result

    expected_keys = set()
    for problem in PROBLEMS:
        for layers in LAYERS:
            for chi, qubits, entanglement in COLUMNS:
                if layers == 1 and entanglement == "entangled":
                    continue
                expected_keys.add((chi, problem, qubits, layers, entanglement))

    missing = sorted(expected_keys - set(lookup))
    unexpected = sorted(set(lookup) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"Result grid mismatch. Missing={missing}; unexpected={unexpected}"
        )
    return lookup


def write_csv(results: list[Result]) -> None:
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "problem",
                "cost_function",
                "qubits",
                "layers",
                "entanglement",
                "seed",
                "test_accuracy",
                "summary_path",
                "image_path",
            ]
        )
        for result in sorted(
            results,
            key=lambda item: (
                list(PROBLEMS).index(item.problem),
                item.layers,
                item.chi,
                item.qubits,
                item.entanglement,
            ),
        ):
            writer.writerow(
                [
                    result.problem,
                    result.chi,
                    result.qubits,
                    result.layers,
                    result.entanglement,
                    SEED,
                    f"{result.accuracy:.8f}",
                    result.summary_path.relative_to(ROOT).as_posix(),
                    result.image_path.relative_to(ROOT).as_posix(),
                ]
            )


def latex_accuracy(
    lookup: dict[tuple[str, str, int, int, str], Result],
    problem: str,
    layers: int,
    column: tuple[str, int, str],
) -> str:
    chi, qubits, entanglement = column
    if layers == 1 and entanglement == "entangled":
        return "--"
    result = lookup[(chi, problem, qubits, layers, entanglement)]
    return f"{result.accuracy:.2f}"


def compact_table_tex(
    lookup: dict[tuple[str, str, int, int, str], Result],
    problem: str,
) -> str:
    chinese_name, label_slug, _ = PROBLEMS[problem]
    lines = [
        rf"    \caption{{{chinese_name}的测试准确率}}",
        rf"    \label{{tab:appendix-{label_slug}}}",
        r"    \setlength{\tabcolsep}{3.2pt}",
        r"    \renewcommand{\arraystretch}{0.78}",
        r"    \resizebox{0.94\textwidth}{!}{%",
        r"    \begin{tabular}{c*{8}{c}}",
        r"        \toprule",
        r"        \multirow{2}{*}{层数} & \multicolumn{3}{c}{$\chi_f^2$} & \multicolumn{5}{c}{$\chi_{wf}^2$} \\",
        r"        \cmidrule(lr){2-4}\cmidrule(lr){5-9}",
        r"        & 1q & \shortstack{2q\\无纠缠} & \shortstack{2q\\有纠缠}",
        r"        & 1q & \shortstack{2q\\无纠缠} & \shortstack{2q\\有纠缠}",
        r"        & \shortstack{4q\\无纠缠} & \shortstack{4q\\有纠缠} \\",
        r"        \midrule",
    ]
    for layers in LAYERS:
        values = [
            latex_accuracy(lookup, problem, layers, column) for column in COLUMNS
        ]
        lines.append(
            "        "
            + str(layers)
            + " & "
            + " & ".join(values)
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
        ]
    )
    return "\n".join(lines)


def grouped_tables_tex(
    lookup: dict[tuple[str, str, int, int, str], Result],
) -> str:
    groups = (
        ("non convex", "crown", "sphere"),
        ("squares", "wavy lines"),
    )
    blocks: list[str] = []
    for group in groups:
        blocks.extend(
            [
                r"\begin{table}[p]",
                r"    \centering",
                r"    \scriptsize",
            ]
        )
        for index, problem in enumerate(group):
            blocks.append(compact_table_tex(lookup, problem))
            if index != len(group) - 1:
                blocks.append(r"    \vspace{0.4em}")
        blocks.append(r"\end{table}")
        blocks.append("")
    return "\n".join(blocks)


def architecture_text(result: Result) -> str:
    chi = r"$\chi_f^2$" if result.chi == "fidelity_chi" else r"$\chi_{wf}^2$"
    entanglement = "有纠缠" if result.entanglement == "entangled" else "无纠缠"
    return f"{chi}, {result.qubits} 量子比特, {entanglement}, {result.layers} 层"


def wilson_interval(
    accuracy: float,
    *,
    sample_size: int = 4000,
    z: float = 1.96,
) -> tuple[float, float]:
    denominator = 1.0 + z**2 / sample_size
    center = (accuracy + z**2 / (2 * sample_size)) / denominator
    half_width = (
        z
        * math.sqrt(
            accuracy * (1.0 - accuracy) / sample_size
            + z**2 / (4 * sample_size**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def result_lookup(
    results: list[Result],
) -> dict[tuple[str, int, int, str, str], Result]:
    return {
        (
            result.problem,
            result.qubits,
            result.layers,
            result.entanglement,
            result.chi,
        ): result
        for result in results
    }


def paired_objective_deltas(
    results: list[Result],
    problem: str | None = None,
) -> list[float]:
    lookup = result_lookup(results)
    deltas: list[float] = []
    selected_problems = (problem,) if problem is not None else tuple(PROBLEMS)
    for selected_problem in selected_problems:
        for qubits, entanglement in (
            (1, "not_entangled"),
            (2, "not_entangled"),
            (2, "entangled"),
        ):
            for layers in LAYERS:
                if layers == 1 and entanglement == "entangled":
                    continue
                weighted = lookup[
                    (
                        selected_problem,
                        qubits,
                        layers,
                        entanglement,
                        "weighted_fidelity_chi",
                    )
                ].accuracy
                ordinary = lookup[
                    (
                        selected_problem,
                        qubits,
                        layers,
                        entanglement,
                        "fidelity_chi",
                    )
                ].accuracy
                deltas.append(weighted - ordinary)
    return deltas


def paired_entanglement_deltas(
    results: list[Result],
    problem: str | None = None,
) -> list[float]:
    lookup = result_lookup(results)
    deltas: list[float] = []
    selected_problems = (problem,) if problem is not None else tuple(PROBLEMS)
    for selected_problem in selected_problems:
        for chi, qubits_values in (
            ("fidelity_chi", (2,)),
            ("weighted_fidelity_chi", (2, 4)),
        ):
            for qubits in qubits_values:
                for layers in LAYERS:
                    if layers == 1:
                        continue
                    entangled = lookup[
                        (
                            selected_problem,
                            qubits,
                            layers,
                            "entangled",
                            chi,
                        )
                    ].accuracy
                    separable = lookup[
                        (
                            selected_problem,
                            qubits,
                            layers,
                            "not_entangled",
                            chi,
                        )
                    ].accuracy
                    deltas.append(entangled - separable)
    return deltas


def nonmonotonic_statistics(results: list[Result]) -> tuple[int, int, int, float]:
    series: dict[tuple[str, str, int, str], list[tuple[int, float]]] = {}
    for result in results:
        key = (
            result.problem,
            result.chi,
            result.qubits,
            result.entanglement,
        )
        series.setdefault(key, []).append((result.layers, result.accuracy))
    nonmonotonic = 0
    decreases: list[float] = []
    transition_count = 0
    for values in series.values():
        ordered = sorted(values)
        changes = [
            ordered[index][1] - ordered[index - 1][1]
            for index in range(1, len(ordered))
        ]
        transition_count += len(changes)
        negative = [change for change in changes if change < 0.0]
        if negative:
            nonmonotonic += 1
            decreases.extend(negative)
    return nonmonotonic, len(decreases), transition_count, min(decreases)


def worst_depth_drop_pair(results: list[Result]) -> tuple[Result, Result]:
    """Return the adjacent-depth pair with the largest accuracy decrease."""
    series: dict[tuple[str, str, int, str], list[Result]] = {}
    for result in results:
        key = (
            result.problem,
            result.chi,
            result.qubits,
            result.entanglement,
        )
        series.setdefault(key, []).append(result)

    worst_pair: tuple[Result, Result] | None = None
    worst_delta = 0.0
    for values in series.values():
        ordered = sorted(values, key=lambda result: result.layers)
        for before, after in zip(ordered, ordered[1:]):
            delta = after.accuracy - before.accuracy
            if delta < worst_delta:
                worst_delta = delta
                worst_pair = (before, after)

    if worst_pair is None:
        raise RuntimeError("No adjacent-depth accuracy decrease was found")
    return worst_pair


def write_statistics_csv(results: list[Result], best: dict[str, Result]) -> None:
    rows: list[dict[str, str | int | float]] = []
    for problem, result in best.items():
        lower, upper = wilson_interval(result.accuracy)
        rows.append(
            {
                "metric": "best_accuracy",
                "scope": problem,
                "n": 4000,
                "value": result.accuracy,
                "lower": lower,
                "upper": upper,
            }
        )
    for problem in (None, *PROBLEMS):
        scope = "all" if problem is None else problem
        objective = paired_objective_deltas(results, problem)
        rows.append(
            {
                "metric": "weighted_minus_ordinary_mean",
                "scope": scope,
                "n": len(objective),
                "value": float(np.mean(objective)),
                "lower": "",
                "upper": "",
            }
        )
        entanglement = paired_entanglement_deltas(results, problem)
        rows.append(
            {
                "metric": "entangled_minus_separable_mean",
                "scope": scope,
                "n": len(entanglement),
                "value": float(np.mean(entanglement)),
                "lower": "",
                "upper": "",
            }
        )
    with STATISTICS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("metric", "scope", "n", "value", "lower", "upper"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_layer_trend_figure(results: list[Result]) -> None:
    lookup = result_lookup(results)
    ordinary_means: list[float] = []
    weighted_means: list[float] = []
    threshold_rates: list[float] = []
    for layers in LAYERS:
        for chi, destination in (
            ("fidelity_chi", ordinary_means),
            ("weighted_fidelity_chi", weighted_means),
        ):
            values: list[float] = []
            for problem in PROBLEMS:
                for qubits, entanglement in (
                    (1, "not_entangled"),
                    (2, "not_entangled"),
                    (2, "entangled"),
                ):
                    if layers == 1 and entanglement == "entangled":
                        continue
                    values.append(
                        lookup[
                            (problem, qubits, layers, entanglement, chi)
                        ].accuracy
                    )
            destination.append(float(np.mean(values)))
        available = [
            result.accuracy for result in results if result.layers == layers
        ]
        threshold_rates.append(float(np.mean(np.asarray(available) >= 0.90)))

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(
        LAYERS,
        ordinary_means,
        "o-",
        linewidth=2,
        label="ordinary fidelity",
    )
    axes[0].plot(
        LAYERS,
        weighted_means,
        "s-",
        linewidth=2,
        label="weighted fidelity",
    )
    axes[0].set_xlabel("Re-uploading layers")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("(a) Shared 1q/2q configurations")

    axes[1].plot(
        LAYERS,
        np.asarray(threshold_rates) * 100.0,
        "o-",
        color="#7a3e9d",
        linewidth=2,
    )
    axes[1].set_xlabel("Re-uploading layers")
    axes[1].set_ylabel("Configurations at or above 90%")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("(b) All available configurations")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "layer_trends.png", dpi=220)
    plt.close(figure)


def copy_best_images(results: list[Result]) -> dict[str, Result]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    best: dict[str, Result] = {}
    for problem in PROBLEMS:
        candidates = [result for result in results if result.problem == problem]
        candidates.sort(
            key=lambda item: (
                -item.accuracy,
                item.qubits,
                item.layers,
                item.entanglement == "entangled",
                item.chi == "weighted_fidelity_chi",
            )
        )
        best[problem] = candidates[0]

    for problem in ("non convex", "crown", "squares", "wavy lines"):
        result = best[problem]
        if not result.image_path.exists():
            raise FileNotFoundError(result.image_path)
        destination = FIGURE_DIR / f"{PROBLEMS[problem][2]}.png"
        shutil.copy2(result.image_path, destination)

    comparison_specs = {
        "sphere_best.png": (
            "sphere",
            4,
            2,
            "entangled",
            "weighted_fidelity_chi",
        ),
        "nonconv_depth_l1.png": (
            "non convex",
            2,
            1,
            "not_entangled",
            "weighted_fidelity_chi",
        ),
        "nonconv_depth_l3.png": (
            "non convex",
            2,
            3,
            "not_entangled",
            "weighted_fidelity_chi",
        ),
        "nonconv_depth_l8.png": (
            "non convex",
            2,
            8,
            "not_entangled",
            "weighted_fidelity_chi",
        ),
        "crown_4q_l2_separable.png": (
            "crown",
            4,
            2,
            "not_entangled",
            "weighted_fidelity_chi",
        ),
        "crown_4q_l2_entangled.png": (
            "crown",
            4,
            2,
            "entangled",
            "weighted_fidelity_chi",
        ),
        "squares_l5_ordinary.png": (
            "squares",
            2,
            5,
            "entangled",
            "fidelity_chi",
        ),
        "squares_l5_weighted.png": (
            "squares",
            2,
            5,
            "entangled",
            "weighted_fidelity_chi",
        ),
    }
    lookup = result_lookup(results)
    for filename, key in comparison_specs.items():
        source = lookup[key].image_path
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIGURE_DIR / filename)

    drop_before, drop_after = worst_depth_drop_pair(results)
    for filename, result in (
        ("worst_depth_drop_before.png", drop_before),
        ("worst_depth_drop_after.png", drop_after),
    ):
        if not result.image_path.exists():
            raise FileNotFoundError(result.image_path)
        shutil.copy2(result.image_path, FIGURE_DIR / filename)

    write_layer_trend_figure(results)
    return best


def write_tex(
    results: list[Result],
    lookup: dict[tuple[str, str, int, int, str], Result],
    best: dict[str, Result],
) -> None:
    objective_all = paired_objective_deltas(results)
    entanglement_all = paired_entanglement_deltas(results)
    nonmonotonic, decrease_count, transition_count, worst_drop = (
        nonmonotonic_statistics(results)
    )
    drop_before, drop_after = worst_depth_drop_pair(results)
    drop_problem = PROBLEMS[drop_before.problem][0]
    drop_chi = (
        r"$\chi_f^2$"
        if drop_before.chi == "fidelity_chi"
        else r"$\chi_{wf}^2$"
    )
    drop_entanglement = (
        "有纠缠" if drop_before.entanglement == "entangled" else "无纠缠"
    )

    best_rows: list[str] = []
    for problem in PROBLEMS:
        result = best[problem]
        lower, upper = wilson_interval(result.accuracy)
        chi = r"$\chi_f^2$" if result.chi == "fidelity_chi" else r"$\chi_{wf}^2$"
        entanglement = "有" if result.entanglement == "entangled" else "无"
        best_rows.append(
            f"        {PROBLEMS[problem][0]} & "
            f"{100 * result.accuracy:.2f}\\% & "
            f"[{100 * lower:.2f}\\%, {100 * upper:.2f}\\%] & "
            f"{chi} & {result.qubits} & {entanglement} & {result.layers} \\\\"
        )

    objective_rows: list[str] = []
    for problem in PROBLEMS:
        deltas = paired_objective_deltas(results, problem)
        objective_rows.append(
            f"        {PROBLEMS[problem][0]} & {100 * np.mean(deltas):+.2f} & "
            f"{100 * np.median(deltas):+.2f} & "
            f"{sum(delta > 0 for delta in deltas)}/23 & "
            f"{sum(delta < 0 for delta in deltas)}/23 \\\\"
        )
    objective_rows.append(
        f"        总体 & {100 * np.mean(objective_all):+.2f} & "
        f"{100 * np.median(objective_all):+.2f} & "
        f"{sum(delta > 0 for delta in objective_all)}/115 & "
        f"{sum(delta < 0 for delta in objective_all)}/115 \\\\"
    )

    sections = [
        r"% This file is generated by experiments/export_appendix_report.py.",
        r"% Do not edit table values by hand; regenerate them from the result summaries.",
        r"\section{论文附录实验结果复现}",
        r"\label{sec:appendix-reproduction}",
        "",
        r"\subsection{实验协议与统计口径}",
        (
            "本节复现原论文附录 B 中的五类分类任务, 共汇总 305 组冻结模型. "
            "全部数值均由普通保真度损失与加权保真度损失目录下的 "
            rf"\texttt{{appendix\_seed\_{SEED}\_summary.txt}} 自动汇总得到. "
            "二维任务使用 200 个训练点和 4000 个测试点, 三维球任务使用 "
            "500 个训练点和 4000 个测试点; 优化器均为 L-BFGS-B. "
            "每个配置只使用随机种子 30 训练一次, 因此本节比较的是固定初始化下的"
            "描述性结果, 不能代替多随机种子的稳定性评估. 完整表按论文格式保留两位"
            "小数, 后续统计均使用未舍入的原始准确率."
        ),
        "",
        (
            "表~\\ref{tab:appendix-best-summary} 同时给出把 4000 个测试样本视为 "
            "Bernoulli 试验时的 95\\% Wilson 区间. 该区间仅刻画固定测试集上的"
            "有限样本波动, 未计入训练初始化、模型选择和超参数搜索的不确定性; "
            "尤其每个任务的最佳值是从 61 个配置中选出的, 因而不能把区间解释为"
            "对整个训练流程的严格置信区间."
        ),
        "",
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{五类任务的最佳配置及固定测试集 Wilson 区间}",
        r"    \label{tab:appendix-best-summary}",
        r"    \small",
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        任务 & 最高准确率 & 95\% Wilson 区间 & 损失 & 比特数 & 纠缠 & 层数 \\",
        r"        \midrule",
        *best_rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
        r"\subsection{完整准确率网格}",
        (
            r"表~\ref{tab:appendix-non-convex}--\ref{tab:appendix-wavy-lines} "
            "给出全部 305 组准确率. 非凸边界任务由 "
            r"$x_2=-2x_1+1.5\sin(\pi x_1)$ 划分类别; 圆环任务要求识别两个"
            "同心圆之间的不连通区域; 三维球任务检验三维输入编码; 方块任务是"
            "四象限线性边界; 波浪线任务同时包含四分类、非凸边界和面积较小的局部区域. "
            "一层电路尚未插入 CZ 门, 因而对应的有纠缠项记为 ``--''. 为避免"
            "一页只出现一张表, 五张完整表按三张和两张成组排版."
        ),
        "",
        grouped_tables_tex(lookup),
        r"\clearpage",
        "",
        r"\subsection{层数效应与饱和区间}",
        (
            "图~\\ref{fig:appendix-layer-trends} 左图只比较普通与加权损失共同拥有的"
            "1q、2q 无纠缠和 2q 有纠缠配置, 避免把加权损失独有的 4q 配置混入"
            "损失函数比较. 在这组共享架构上, 普通保真度平均准确率从一层的 55.01\\%"
            "上升到十层的 92.63\\%, 加权保真度则从 66.25\\% 上升到 93.56\\%. "
            "右图使用每层全部可用配置: 达到 90\\% 的比例从一层的 4\\% 增至五层的"
            "70\\%, 六层和八层均为 90\\%, 十层为 95\\%. "
            "与此同时, 全配置平均准确率在五、六、八、十层分别为 91.98\\%、"
            "93.27\\%、93.56\\% 和 93.28\\%, 表明主要增益集中在前五至六层, "
            "此后总体进入平台区."
        ),
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.57\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/layer_trends.png}",
        r"        \caption{共享架构平均准确率与达到 90\% 的配置比例}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.40\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/sphere_best.png}",
        r"        \caption{三维球最佳模型的 $xy$ 投影, 准确率 96.17\%}",
        r"    \end{subfigure}",
        r"    \caption{层数总体趋势与此前未展示的三维球结果}",
        r"    \label{fig:appendix-layer-trends}",
        r"\end{figure}",
        "",
        (
            f"逐架构检查进一步表明, 40 条层数序列中有 {nonmonotonic} 条至少出现一次"
            f"随层数增加而下降; 全部 {transition_count} 个相邻层数转换中有 "
            f"{decrease_count} 个下降, 最大单步下降为 {abs(100 * worst_drop):.2f} "
            "个百分点. 最大下降实例出现在"
            f"{drop_problem}的 {drop_chi}、{drop_before.qubits}q、"
            f"{drop_entanglement}架构: 从 {drop_before.layers} 层的 "
            f"{100 * drop_before.accuracy:.2f}\\% 降至 {drop_after.layers} 层的 "
            f"{100 * drop_after.accuracy:.2f}\\%. 图~"
            "\\ref{fig:appendix-worst-depth-drop} 显示, 较深模型并非缺少表达该边界的"
            "能力, 而是本次独立优化得到的决策区域退化为近似水平分割, 在正弦边界的"
            "弯曲区域产生大量错误. 因而“更深通常更好”只在总体平均意义上成立, "
            "不能作为单个训练实例的单调性结论. 该现象与非凸优化陷入较差局部极小值"
            "相符, 但仅凭单随机种子仍不能把下降唯一归因于局部极小值; 还需要多随机"
            "种子和优化轨迹验证."
        ),
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/worst_depth_drop_before.png}",
        (
            rf"        \caption{{{drop_before.layers} 层: "
            rf"{100 * drop_before.accuracy:.2f}\%}}"
        ),
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/worst_depth_drop_after.png}",
        (
            rf"        \caption{{{drop_after.layers} 层: "
            rf"{100 * drop_after.accuracy:.2f}\%}}"
        ),
        r"    \end{subfigure}",
        (
            rf"    \caption{{最大相邻层数下降实例: {drop_problem}, {drop_chi}, "
            rf"{drop_before.qubits}q, {drop_entanglement}; 增加一层后下降 "
            rf"{100 * (drop_before.accuracy - drop_after.accuracy):.2f} 个百分点}}"
        ),
        r"    \label{fig:appendix-worst-depth-drop}",
        r"\end{figure}",
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/nonconv_depth_l1.png}",
        r"        \caption{1 层: 75.58\%}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/nonconv_depth_l3.png}",
        r"        \caption{3 层: 94.93\%}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.32\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/nonconv_depth_l8.png}",
        r"        \caption{8 层: 97.75\%}",
        r"    \end{subfigure}",
        r"    \caption{同一 $\chi_{wf}^2$、2q 无纠缠架构随重上传层数增加的决策区域}",
        r"    \label{fig:appendix-depth-progression}",
        r"\end{figure}",
        "",
        r"\subsection{损失函数的配对比较}",
        (
            "为避免 4q 配置只存在于加权损失一侧造成结构混杂, 表~"
            "\\ref{tab:appendix-objective-pairs} 仅在相同任务、比特数、层数和纠缠设置"
            "下比较两种损失, 每个任务含 23 对, 总计 115 对. 加权保真度的平均增益为"
            f" {100 * np.mean(objective_all):.2f} 个百分点, 中位增益为 "
            f"{100 * np.median(objective_all):.2f} 个百分点, 并在 "
            f"{sum(delta > 0 for delta in objective_all)}/115 个配对中占优. "
            "但方块任务的平均增益只有 0.77 个百分点, 且仅 8/23 个配对占优, "
            "说明加权损失并非对所有几何边界都稳定更好."
        ),
        "",
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{共享架构上加权保真度相对普通保真度的配对差值}",
        r"    \label{tab:appendix-objective-pairs}",
        r"    \small",
        r"    \begin{tabular}{lrrrr}",
        r"        \toprule",
        r"        任务 & 平均差值/百分点 & 中位差值/百分点 & 加权胜出 & 普通胜出 \\",
        r"        \midrule",
        *objective_rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
        r"\begin{figure}[htbp]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/squares_l5_ordinary.png}",
        r"        \caption{$\chi_f^2$: 98.58\%}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/squares_l5_weighted.png}",
        r"        \caption{$\chi_{wf}^2$: 93.60\%}",
        r"    \end{subfigure}",
        r"    \caption{方块任务中同为 2q、有纠缠、5 层时两种损失的结果对照}",
        r"    \label{fig:appendix-loss-comparison}",
        r"\end{figure}",
        "",
        r"\subsection{纠缠效应的异质性}",
        (
            "在固定任务、损失、比特数和层数后, 共得到 105 对有纠缠与无纠缠结果. "
            f"有纠缠模型平均只提高 {100 * np.mean(entanglement_all):.2f} 个百分点, "
            f"中位数为 {100 * np.median(entanglement_all):.2f} 个百分点; "
            f"{sum(delta > 0 for delta in entanglement_all)} 对提高, "
            f"{sum(delta < 0 for delta in entanglement_all)} 对下降, "
            f"差值范围为 {100 * min(entanglement_all):.2f} 至 "
            f"{100 * max(entanglement_all):.2f} 个百分点. "
            "因此不能仅凭平均值断言纠缠必然改善分类. 图~"
            "\\ref{fig:appendix-entanglement-comparison} 展示收益最大的圆环实例: "
            "4q、2 层加权模型加入 CZ 后由 70.45\\% 提升至 96.00\\%, "
            "说明纠缠的价值主要取决于任务结构与当前深度."
        ),
        "",
        r"\begin{figure}[H]",
        r"    \centering",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/crown_4q_l2_separable.png}",
        r"        \caption{无纠缠: 70.45\%}",
        r"    \end{subfigure}",
        r"    \hfill",
        r"    \begin{subfigure}[t]{0.48\textwidth}",
        r"        \centering",
        r"        \includegraphics[width=\linewidth]{figures/appendix_results/crown_4q_l2_entangled.png}",
        r"        \caption{有纠缠: 96.00\%}",
        r"    \end{subfigure}",
        r"    \caption{圆环任务中同为 $\chi_{wf}^2$、4q、2 层时的纠缠对照}",
        r"    \label{fig:appendix-entanglement-comparison}",
        r"\end{figure}",
        "",
        r"\subsection{各二维任务的最佳决策区域}",
        (
            "图~\\ref{fig:appendix-best-results} 汇总四个二维任务的最佳结果. "
            "每幅图左侧为预测类别, 右侧以绿色和红色标记正确与错误测试点. "
            "这些图用于展示最佳可达边界, 而图~\\ref{fig:appendix-worst-depth-drop}--"
            "\\ref{fig:appendix-entanglement-comparison} 则补充了原论文未集中展示的"
            "深度、损失与纠缠对照."
        ),
        "",
        r"\begin{figure}[p]",
        r"    \centering",
    ]
    for index, problem in enumerate(("non convex", "crown", "squares", "wavy lines")):
        chinese_name, _, file_slug = PROBLEMS[problem]
        result = best[problem]
        sections.extend(
            [
                r"    \begin{subfigure}[t]{0.48\textwidth}",
                r"        \centering",
                rf"        \includegraphics[width=\linewidth]{{figures/appendix_results/{file_slug}.png}}",
                (
                    rf"        \caption{{{chinese_name}: {architecture_text(result)}, "
                    rf"准确率 {100 * result.accuracy:.2f}\%}}"
                ),
                rf"        \label{{fig:appendix-{file_slug}-best}}",
                r"    \end{subfigure}",
                r"    \hfill" if index % 2 == 0 else r"    \par\medskip",
            ]
        )
    sections.extend(
        [
            r"    \caption{附录四个二维任务的最佳复现实验结果}",
            r"    \label{fig:appendix-best-results}",
            r"\end{figure}",
            "",
            r"\enlargethispage{2\baselineskip}",
            r"\subsection{结论与限制}",
            (
                "305 组结果支持三个有限结论. 第一, 重上传层数是最稳定的容量来源, "
                "平均性能在五至六层后趋于饱和, 但单次训练结果高度非单调. 第二, 在严格"
                "匹配的共享架构上, 加权保真度总体优于普通保真度, 但优势在方块任务上"
                "明显减弱, 最佳方块模型反而来自普通保真度. 第三, 纠缠的总体平均收益"
                "很小且正负并存, 其作用更接近任务相关的表达能力补充, 而不是普遍增益. "
                "以上结果均来自单一训练种子; 若要作统计推断, 仍需对每个配置使用多个"
                "独立初始化, 报告均值、标准差或置信区间, 并对同一测试集上的配对差值"
                "采用配对重采样或 McNemar 检验."
            ),
            "",
        ]
    )
    TEX_PATH.write_text("\n".join(sections), encoding="utf-8")


def sync_main_tex() -> None:
    """Embed the generated Section 10 source directly in the main TeX file."""
    generated = TEX_PATH.read_text(encoding="utf-8").rstrip()
    main_tex = MAIN_TEX_PATH.read_text(encoding="utf-8")
    block = (
        f"{MAIN_TEX_BEGIN_MARKER}\n"
        f"{generated}\n"
        f"{MAIN_TEX_END_MARKER}"
    )

    if MAIN_TEX_BEGIN_MARKER in main_tex and MAIN_TEX_END_MARKER in main_tex:
        start = main_tex.index(MAIN_TEX_BEGIN_MARKER)
        end = main_tex.index(MAIN_TEX_END_MARKER, start)
        end += len(MAIN_TEX_END_MARKER)
        main_tex = main_tex[:start] + block + main_tex[end:]
    elif r"\input{appendix_experiment_results}" in main_tex:
        main_tex = main_tex.replace(
            r"\input{appendix_experiment_results}",
            block,
            1,
        )
    else:
        raise RuntimeError(
            "Could not locate the Section 10 generated block in "
            f"{MAIN_TEX_PATH}"
        )

    MAIN_TEX_PATH.write_text(main_tex, encoding="utf-8")


def main() -> None:
    results = read_results()
    lookup = validate(results)
    write_csv(results)
    best = copy_best_images(results)
    write_statistics_csv(results, best)
    write_tex(results, lookup, best)
    sync_main_tex()
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {STATISTICS_CSV_PATH}")
    print(f"Wrote {TEX_PATH}")
    print(f"Updated embedded Section 10 in {MAIN_TEX_PATH}")
    for problem, result in best.items():
        print(
            f"{problem}: accuracy={result.accuracy:.5f}, chi={result.chi}, "
            f"qubits={result.qubits}, layers={result.layers}, "
            f"entanglement={result.entanglement}"
        )


if __name__ == "__main__":
    main()
