# M2：Qubit、Depth 与 CZ 资源对照

Canonical summary：
[`20260728T120613.324618Z/`](20260728T120613.324618Z/)

M2 新增 15 次 optimizer runs，并复用 M1 的五个 1Q-L4 squared
checkpoints。

| 模型 | Parameters | Template CZ | Mean test accuracy | Mean nfev |
| --- | ---: | ---: | ---: | ---: |
| 1Q-L4 | 20 | 0 | 0.83630 | 3045.0 |
| 1Q-L2 | 10 | 0 | 0.78695 | 288.2 |
| 2Q-L2 separable | 20 | 0 | 0.51600 | 315.0 |
| 2Q-L2 CZ | 20 | 1 | 0.60485 | 714.0 |

已知负结果：2Q-CZ seed 31 在 Qiskit level 3 的 13 行概率误差超过严格
`1e-10` 阈值，最大为 `4.6520e-10`；全部 4000 行的预测标签仍一致。该 finding
和独立诊断均保留，不能表述为 probability parity 已通过。

Raw 根目录：[`../../raw/M2/`](../../raw/M2/)
