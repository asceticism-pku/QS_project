# M3：Re-uploading Layer Pruning

Canonical summary：
[`20260728T121800.547650Z/`](20260728T121800.547650Z/)

M3 新增 15 次 optimizer runs，并复用 M1 的五个 L4 baseline checkpoints。

| 模型 | Parameters | Mean test accuracy | Own nfev | Pipeline nfev |
| --- | ---: | ---: | ---: | ---: |
| L4 baseline | 20 | 0.83630 | 3045.0 | 3045.0 |
| L4→L3 selected pruning | 15 | 0.82400 | 412.8 | 3457.8 |
| L4 truncate last | 15 | 0.82290 | 476.8 | 3521.8 |
| L3 scratch | 15 | 0.79915 | 838.4 | 838.4 |

Pruned 相对 baseline 的 paired mean accuracy delta 为 `-0.01230`，超过允许的
`0.005` 降幅，因此 M3 判定为 negative。层数、参数与 level-0 median depth
均减少 25%，exact parity 通过；level-3 median depth 仍为 5，不能声称 native
gate depth 获得优化。

Raw 根目录：[`../../raw/M3/`](../../raw/M3/)
