# Code for *[Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085)*
#### Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre.

This is a repository for all code written for the article "*Data re-uploading for a universal quantum classifier*. Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre."
It gives numerical simulations of the quantum classifier in [Quantum 4, 226 (2020)](https://quantum-journal.org/papers/q-2020-02-06-226/).

All code is written Python. Libraries required:
  - matplotlib for plots
  - numpy, os, scipy
  - scikit-learn

##### Files included:
  - [QuantumState.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/QuantumState.py): Simulator of a quantum circuit using only basic Python packages such as numpy
  - [big_functions.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/big_functions.py): Functions acting as the master of all other subroutines in the simulator
  - [circuitery.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/circuitery.py): Translates the problem to the quantum circuit basic level.
  - [classical_benchmark.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/classical_benchmark.py): Provides some classical examples using scikit learn.
  - [data_gen.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/data_gen.py): Generates random training and data set for different problems.
  - [fidelity_minimization.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/fidelity_minimization.py): All the code needed for the fidelity cost function.
  - [main.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/main.py): This is the only file one needs to change. Everything can be set up there: number of qubits, layers, entanglement, cost function, problem, etc. The only thing one has to do is to run this file.
  - [problem__gen.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/problem_gen.py): Generates data of the problem we need for other files.
  - [save_data.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/save_data.py): Saves results in text files and images. 
  - [test_data.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/test_data.py): Tests the performance of the classifier, and outputs variables needed for saving data.
  - [weighted_fidelity_minimization.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/weighted_fidelity_minimization.py): All the code needed for the weighted fidelity cost function.
##### How to cite
If you use this code in your research, please cite it as follows:

Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. Quantum, 4, 226.

BibTeX:
```
@article{P_rez_Salinas_2020,
   title={Data re-uploading for a universal quantum classifier},
   volume={4},
   ISSN={2521-327X},
   url={http://dx.doi.org/10.22331/q-2020-02-06-226},
   DOI={10.22331/q-2020-02-06-226},
   journal={Quantum},
   publisher={Verein zur Forderung des Open Access Publizierens in den Quantenwissenschaften},
   author={Pérez-Salinas, Adrián and Cervera-Lierta, Alba and Gil-Fuster, Elies and Latorre, José I.},
   year={2020},
   month={Feb},
   pages={226}
}

```

## Course project extension

This repository also contains the team's reproducible circle-dataset experiments
for the course project. The extension keeps the original simulator files
unchanged and adds thin wrappers, tests, raw artifacts, summaries, compilation
audits, layer pruning, and adaptive-shot evaluation.

- [Experiment contract, commands, and verified results](PROJECT_EXPERIMENTS.md)
- [`experiments/`](experiments/): command-line runners and artifact verifiers
- [`src/qs_project/`](src/qs_project/): project-specific adapters and evaluation
  logic
- [`tests/`](tests/): regression and contract tests
- [`results/`](results/): immutable raw artifacts and derived CSV/JSON summaries
- [`doc/大作业报告/`](doc/大作业报告/): collaborative LaTeX report

The formal training matrix contains exactly 45 optimizer runs. The M4
fixed/adaptive-shot comparison reuses ten frozen checkpoints and performs no
additional optimization.
# 含噪声模拟与 Origin Quantum Cloud

附录实验保存的冻结参数现在可以直接用于有限 shots、门噪声和云端模拟器评估。默认命令测试 `crown` 数据集上表现最好的
`weighted_fidelity_chi / 1 qubit / 10 layers` 模型，并生成逐点结果、噪声扫描表和准确率曲线：

```powershell
python experiments/run_noisy_cloud_evaluation.py --backend local
```

本地模型使用单/双量子比特退极化噪声和对称读出噪声。`--noise-levels` 指定单比特门错误率；默认双比特错误率为其 10 倍，读出错误率为其 2 倍。加权保真度模型会执行 X/Y/Z 三组测量，以保持与论文约化密度矩阵分类判据一致。

提交到 Origin Quantum Cloud 前，安装可选 SDK 并在当前终端设置 API Key：

```powershell
python -m pip install -r requirements-cloud.txt
$env:QPANDA_QCLOUD_API_KEY="你的 API Key"
python experiments/run_noisy_cloud_evaluation.py --list-origin-backends
python experiments/run_noisy_cloud_evaluation.py --backend origin-cloud --origin-backend full_amplitude --points 40 --noise-levels 0.001 --readout-multiplier 0
```

云端后端是否可用取决于账户和平台实时状态，请先运行 `--list-origin-backends`。每个已提交批次的 job ID 会立即追加保存到输出目录的 `origin_job_ids.jsonl`，避免长任务中断后丢失任务编号。
