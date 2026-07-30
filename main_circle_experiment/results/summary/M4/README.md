# M4：Fixed 与 Adaptive Shots

Canonical summary：
[`20260728T122358.008884Z/`](20260728T122358.008884Z/)

`20260728T122236.033739Z/` 是保留的较早汇总，不作为报告的 canonical source。

M4 使用 M1 的五个 L4 baseline 与 M3 的五个 pruned checkpoints：

- optimizer runs：0
- nfev：0
- evaluation points：1000（每类 500）
- repeats：每 checkpoint 100
- campaigns：1000

| 模型 | 方法 | Mean accuracy | Mean shots |
| --- | --- | ---: | ---: |
| L4 | Fixed 128 | 0.819440 | 128.000 |
| L4 | Fixed 512 | 0.821110 | 512.000 |
| L4 | Fixed 2048 | 0.821892 | 2048.000 |
| L4 | Adaptive | 0.821848 | 315.228 |
| Pruned L3 | Fixed 128 | 0.804662 | 128.000 |
| Pruned L3 | Fixed 512 | 0.805934 | 512.000 |
| Pruned L3 | Fixed 2048 | 0.806382 | 2048.000 |
| Pruned L3 | Adaptive | 0.806396 | 329.777 |

两个模型的 adaptive 策略都通过预设判据。因为 M3 pruning 本身未通过准确率
条件，不能据此声称联合 layer+shots 优化成功。

Raw 根目录：[`../../raw/M4/`](../../raw/M4/)
