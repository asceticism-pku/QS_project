import itertools

from big_functions import (
    minimizer,
    paint_world,
    painter,
)


def main():
    problems = [
        "non convex",
        "crown",
        "sphere",
        "squares",
        "wavy lines",
    ]
    qubits_list = [1, 2, 4]
    layers_list = [1, 2, 3, 4, 5, 6, 8, 10]
    entanglement_list = [False, True]
    chi_list = [
        "fidelity_chi",
        "weighted_fidelity_chi",
    ]
    method = "L-BFGS-B"
    seed = 30
    draw_decision_boundary = True
    draw_bloch_world = False
    experiments = []

    for problem, qubits, layers, entanglement, chi in itertools.product(
        problems,
        qubits_list,
        layers_list,
        entanglement_list,
        chi_list,
    ):
        if qubits == 1 and entanglement is True:
            continue
        if layers == 1 and entanglement is True:
            continue
        if chi == "fidelity_chi" and qubits == 4:
            continue
        experiments.append((problem, qubits, layers, entanglement, chi))

    total_experiments = len(experiments)
    print(f"开始自动复现附录实验，共计 {total_experiments} 组有效参数组合...")
    failures = []

    for experiment_count, (
        problem,
        qubits,
        layers,
        entanglement,
        chi,
    ) in enumerate(experiments, start=1):
        entanglement_code = "y" if entanglement else "n"
        name = f"appendix_seed_{seed}"

        print()
        print("=" * 80)
        print(
            f"[{experiment_count}/{total_experiments}] "
            f"问题={problem}, "
            f"量子比特={qubits}, "
            f"层数={layers}, "
            f"纠缠={entanglement_code}, "
            f"代价函数={chi}"
        )

        try:
            minimizer(
                chi,
                problem,
                qubits,
                entanglement_code,
                layers,
                method,
                name,
                seed=seed,
            )

            if draw_decision_boundary:
                painter(
                    chi,
                    problem,
                    qubits,
                    entanglement_code,
                    layers,
                    method,
                    name,
                    standard_test=True,
                    seed=seed,
                )
            if draw_bloch_world and qubits == 1 and problem != "sphere":
                paint_world(
                    chi,
                    problem,
                    qubits,
                    entanglement_code,
                    layers,
                    method,
                    name,
                    standard_test=True,
                    seed=seed,
                )
            print(f"[{experiment_count}/{total_experiments}] 实验完成。")

        except Exception as error:
            config = {
                "problem": problem,
                "qubits": qubits,
                "layers": layers,
                "entanglement": entanglement_code,
                "chi": chi,
                "error": repr(error),
            }
            failures.append(config)

            print(f"[{experiment_count}/{total_experiments}] 实验失败：{error!r}")
            print("继续运行下一组实验。")

    print()
    print("=" * 80)
    print(
        f"全部实验运行结束：成功 "
        f"{total_experiments - len(failures)} 组，"
        f"失败 {len(failures)} 组。"
    )

    if failures:
        print("\n失败配置如下：")
        for failure in failures:
            print(
                f"- problem={failure['problem']}, "
                f"qubits={failure['qubits']}, "
                f"layers={failure['layers']}, "
                f"entanglement={failure['entanglement']}, "
                f"chi={failure['chi']}, "
                f"error={failure['error']}"
            )


if __name__ == "__main__":
    main()
