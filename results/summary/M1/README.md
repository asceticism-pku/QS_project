# M1：Loss 语义与作者层数趋势

Canonical summary：
[`20260728T120122.559035Z/`](20260728T120122.559035Z/)

M1 共 15 次 optimizer runs：

- 作者 weighted 1Q 层数趋势：L1/L2/L4/L8，seed 30，共 4 次。
- 作者 amplitude 1Q-L4 参考：seed 30，共 1 次。
- Controlled amplitude/squared 1Q-L4：seeds 30--34，共 10 次。

| 对照 | Mean test accuracy | Sample SD | Mean nfev |
| --- | ---: | ---: | ---: |
| `legacy_amplitude` | 0.88440 | 0.02787 | 2520.0 |
| `paper_squared` | 0.83630 | 0.03265 | 3045.0 |

Paired `squared - amplitude = -0.04810`，超过预设的 `0.005` practical-effect
阈值。作者趋势中的 L8 以 `status=1, nfev=15007` 到达原始预算上限；终态未被
删除。

Raw 根目录：[`../../raw/M1/`](../../raw/M1/)
