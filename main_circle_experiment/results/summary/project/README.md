# 全项目一致性审计

Canonical passed audit：
[`20260728T123108.528816Z/`](20260728T123108.528816Z/)

该审计确认：

- optimizer matrix 为 45/45，M1/M2/M3 各 15 次；
- 45 个 optimizer identity 各出现一次；
- 唯一未收敛的 M1 weighted L8 运行被保留；
- M2 的 probability-parity negative 与 artifact integrity 分开记录；
- M3 compile artifacts 与 exact parity 完整；
- M4 精确复用十个冻结 checkpoints，`optimizer_runs=0, nfev=0`；
- `passed=true` 且 `issues=[]`。

`20260728T122844.515995Z/` 是保留的首次失败审计。它把 M2 已知的科学负结果误归
为 artifact failure；后续审计器只修正分类逻辑，没有修改 M2 raw 或阈值。
