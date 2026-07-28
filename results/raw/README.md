# Raw results

原始产物按正文实验 ID 分开：

```text
raw/
├── P0/                         smoke test
├── M1/                         loss 与作者层数趋势
├── M2/                         qubit/depth/CZ 与编译审计
├── M3/                         pruning 与编译审计
└── M4/                         fixed/adaptive shots
```

训练类目录采用：

```text
M{n}/{config-id}/seed-{seed}/{UTC timestamp}-{unique suffix}/
```

每个 optimizer run 至少包含：

- `config.json`
- `command.txt`
- `environment.json`
- `initial_checkpoint.npz`
- `checkpoint.npz`
- `progress.jsonl`
- `result.json`

编译审计、诊断和 finite-shot 评价使用各自的等价 schema。所有 raw 目录均为
append-only：不得覆盖、规范化历史路径或删除负结果。
