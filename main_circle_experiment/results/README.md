# 正文实验结果索引

本目录只保存实验输入索引、不可覆盖的原始产物和由原始产物生成的汇总。正文实验
按 M1--M4 分开存放；论文 Appendix B 的同伴结果仍位于仓库原有的
`fidelity_chi/` 与 `weighted_fidelity_chi/`，不混入本目录。

| 实验 | 内容 | Optimizer runs | Raw | Canonical summary | 状态 |
| --- | --- | ---: | --- | --- | --- |
| M1 | 作者层数趋势、amplitude 参考、五 seed loss 对照 | 15 | [`raw/M1/`](raw/M1/) | [`summary/M1/`](summary/M1/) | `completed-verified` |
| M2 | 1Q-L2、1Q-L4、2Q-L2 separable/CZ | 15 | [`raw/M2/`](raw/M2/) | [`summary/M2/`](summary/M2/) | `completed-negative` |
| M3 | L4 baseline、pruned、truncate-last、L3 scratch | 15 | [`raw/M3/`](raw/M3/) | [`summary/M3/`](summary/M3/) | `completed-negative` |
| M4 | fixed 128/512/2048 与 adaptive shots | 0 | [`raw/M4/`](raw/M4/) | [`summary/M4/`](summary/M4/) | `completed-passed` |

正文训练总数严格为 45 次。M4 只评价十个冻结 checkpoint，不重新训练。

其他入口：

- [`datasets/`](datasets/)：circle 数据集快照。
- [`indices/`](indices/)：M2/M3 编译抽样与 M4 评估抽样索引。
- [`raw/README.md`](raw/README.md)：raw schema 与不可覆盖规则。
- [`summary/project/`](summary/project/)：全项目 45-run 一致性审计。
- [`../PROJECT_EXPERIMENTS.md`](../PROJECT_EXPERIMENTS.md)：完整合同、命令、
  指标与 provenance 边界。

## 使用原则

- 报告数字优先引用各实验 README 标出的 canonical summary。
- 不修改 raw 中的绝对路径、历史 branch、commit 或 dirty-tree 记录。
- 不删除未收敛或未通过阈值的产物。
- M2/M3 的 negative 是实验结论，不是缺失结果。
- 所有新运行和新汇总必须使用新的时间戳目录。
