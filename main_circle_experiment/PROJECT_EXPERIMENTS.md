# Course project experiments

This document is the repository entry point for the course-project extension.
The upstream files implement the simulator from Pérez-Salinas et al.,
*Data re-uploading for a universal quantum classifier*. The project extension
uses those files through thin wrappers; it does not replace the simulator.

## Scope and status

The main experiments are four circle-dataset comparisons. They are not the four
benchmark families in Section 6 of the paper.

| ID | Comparison | Status |
| --- | --- | --- |
| P0 | Circle, 1 qubit, 1 layer, seed 30 smoke test | Verified |
| M1 | Amplitude versus squared-fidelity loss | Verified |
| M2 | 1Q/2Q/CZ accuracy-resource comparison | Completed with a preserved parity finding |
| M3 | L4 baseline, selected pruning, last-layer truncation, and L3 scratch | Completed; pruning accuracy criterion not met |
| M4 | Fixed 128/512/2048 versus adaptive 128→512→2048 shots | Passed |

Appendix B files already present in the repository are collaborator-uploaded
supplementary results. They were not rerun by this workflow and do not replace
M1--M4.

## Fixed experiment contract

- Dataset: circle classification with 200 training and 4000 test points.
- Data seed: 30.
- Dataset hash:
  `d163876f916dd70372aa75291f99cafd4d1ce14aa4e08f911714055b3ffe3a12`.
- Controlled initialization seeds: 30, 31, 32, 33, and 34.
- Optimizer: L-BFGS-B.
- Default budget:
  `maxfun=15000`, `maxiter=15000`, `ftol=2.22e-9`, `gtol=1e-5`.
- L3 scratch budget: `maxfun=maxiter=30000`.
- The test set is used only after parameters and pruning choices are frozen.
- Failed or nonconverged runs remain in `results/raw/`.

The formal optimizer matrix is:

| Stage | Runs |
| --- | ---: |
| M1 | 15 |
| M2 | 15 |
| M3 | 15 |
| M4 | 0 |
| Total | 45 |

The five M1 `1q-l4-paper_squared` checkpoints are reused as the M2 `1q-l4`
baseline and M3 `l4-base`. M4 reuses those five checkpoints and the five M3
`l4-to-l3-pruned` checkpoints without retraining.

## Environment and commands

The recorded environment uses Python 3.11 with the exact package versions in
`requirements-project.txt`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-project.txt
.venv/bin/python -m pytest -q
```

The experiment entry points are:

```bash
.venv/bin/python experiments/run_project.py p0
.venv/bin/python experiments/run_project.py m1 --slice unit
.venv/bin/python experiments/run_project.py m1 --slice trend
.venv/bin/python experiments/run_project.py m1 --slice reference
.venv/bin/python experiments/run_project.py m1 --slice loss
.venv/bin/python experiments/run_project.py m2
.venv/bin/python experiments/run_compile_audit.py m2
.venv/bin/python experiments/run_project.py m3
.venv/bin/python experiments/run_compile_audit.py m3
.venv/bin/python experiments/summarize_m3.py
.venv/bin/python experiments/run_shot_evaluation.py
.venv/bin/python experiments/summarize_m4.py
.venv/bin/python experiments/verify_project.py
```

Runners and summarizers create new timestamped directories. They do not
overwrite existing raw results. To inspect the published run without retraining,
start with the final audit:

```text
results/summary/project/20260728T123108.528816Z/final_audit.json
```

That audit reports `passed=true`, `issues=[]`, 45 unique optimizer identities,
and 246 verified input artifacts. Its checksum manifest is stored alongside it.

## Main results

### M1: loss semantics

The author implementation's unweighted objective minimizes overlap amplitude,
whereas Equation (7) in the paper uses squared fidelity. On the controlled
1Q-L4 comparison:

| Loss | Mean test accuracy | Sample SD | Mean nfev |
| --- | ---: | ---: | ---: |
| Amplitude | 0.88440 | 0.02787 | 2520.0 |
| Squared fidelity | 0.83630 | 0.03265 | 3045.0 |

The mean paired accuracy delta, squared minus amplitude, is -0.04810. Its
absolute magnitude exceeds the predeclared 0.005 practical-effect threshold.
This is a correctness and reproducibility finding, not a resource optimization.

The author-style weighted layer trend reached test accuracies 0.50325, 0.93500,
0.94450, and 0.96750 for L1, L2, L4, and L8. L8 stopped at the original
evaluation limit with `status=1` and `nfev=15007`; its checkpoint and terminal
metrics are preserved.

### M2: qubit, depth, and CZ

| Model | Parameters | Template CZ | Mean test accuracy | Sample SD | Mean nfev |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1Q-L4 | 20 | 0 | 0.83630 | 0.03265 | 3045.0 |
| 1Q-L2 | 10 | 0 | 0.78695 | 0.02303 | 288.2 |
| 2Q-L2 separable | 20 | 0 | 0.51600 | 0.00000 | 315.0 |
| 2Q-L2 CZ | 20 | 1 | 0.60485 | 0.12178 | 714.0 |

None of the alternatives is within the predeclared ±0.005 similarity interval
of 1Q-L4. CZ improves the five-seed mean over the separable 2Q model by 0.08885,
but the gain is driven by two seeds.

All 4000 compiled rows preserve labels. Level-0 probability parity passes. At
Qiskit optimization level 3, 13 rows from one 2Q-CZ checkpoint exceed the strict
`1e-10` tolerance; the maximum error is `4.6520e-10`. The diagnostic and the
failed-threshold evidence are preserved rather than hidden or relaxed.

### M3: layer pruning

| Model | Parameters | Mean test accuracy | Sample SD | Mean own nfev | Mean pipeline nfev |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4 baseline | 20 | 0.83630 | 0.03265 | 3045.0 | 3045.0 |
| L4→L3 selected pruning | 15 | 0.82400 | 0.02571 | 412.8 | 3457.8 |
| L4 truncate last layer | 15 | 0.82290 | 0.02326 | 476.8 | 3521.8 |
| L3 scratch | 15 | 0.79915 | 0.02985 | 838.4 | 838.4 |

Selected pruning reduces layers, parameters, and level-0 median target depth by
25%. Exact compiled parity passes. Its mean paired test-accuracy drop is 0.01230,
which exceeds the allowed 0.005, so the structural-optimization criterion fails.
Level-3 median target depth remains 5 for both L4 and L3, so this is not claimed
as a native-gate depth reduction.

### M4: adaptive shots

| Model | Method | Mean accuracy | Sample SD | Mean shots |
| --- | --- | ---: | ---: | ---: |
| L4 baseline | Fixed 128 | 0.819440 | 0.035172 | 128.000 |
| L4 baseline | Fixed 512 | 0.821110 | 0.034850 | 512.000 |
| L4 baseline | Fixed 2048 | 0.821892 | 0.034409 | 2048.000 |
| L4 baseline | Adaptive | 0.821848 | 0.034451 | 315.228 |
| Pruned L3 | Fixed 128 | 0.804662 | 0.023724 | 128.000 |
| Pruned L3 | Fixed 512 | 0.805934 | 0.023617 | 512.000 |
| Pruned L3 | Fixed 2048 | 0.806382 | 0.022910 | 2048.000 |
| Pruned L3 | Adaptive | 0.806396 | 0.022919 | 329.777 |

Both adaptive variants satisfy the predeclared accuracy and shot thresholds.
Relative to fixed 2048, the paired mean accuracy deltas are -0.000044 for L4
and +0.000014 for pruned L3. M4 uses 1000 evaluation points, 100 campaigns per
checkpoint, and common prefixes from one 2048-shot stream.

Because M3 fails its accuracy criterion, these results do not establish a joint
layer-plus-shot optimization. Adaptive measurement reduction is reported
separately.

## Artifact map

| Artifact | Canonical path |
| --- | --- |
| Dataset | `results/datasets/circle-seed-30-d163876f916dd703.npz` |
| M1 summary | `results/summary/M1/20260728T120122.559035Z/` |
| M2 summary | `results/summary/M2/20260728T120613.324618Z/` |
| M3 summary | `results/summary/M3/20260728T121800.547650Z/` |
| M4 summary | `results/summary/M4/20260728T122358.008884Z/` |
| Final audit | `results/summary/project/20260728T123108.528816Z/` |

Every optimizer run stores its configuration, command, dataset hash, initial
and final checkpoint, metrics, optimizer status, `nfev`, environment, and code
revision. The raw revision records the pre-publication base commit
`0d647b0e8019d3b0ea59baf1af82b51a08bb6448` plus dirty-tree/source hashes,
because the experiments were completed before this extension was committed.
Those historical provenance records must not be rewritten to resemble the
later publication commit.

## Provenance boundary

The result metrics, checkpoint lineage, dataset identity, commands, and
registered artifact hashes are complete and pass the final audit. Two narrower
historical-source limitations remain:

- The P0 `command.txt` records the pyenv Python executable while its
  `environment.json` records the repository virtual-environment executable.
  The artifacts do not establish why the two paths differ.
- The optimizer-run records hash the core training source and record the
  untracked runner path, but they do not contain a contemporaneous standalone
  hash of `experiments/run_project.py`. The first M2 compilation artifact also
  predates the explicit compile-runner/module hash fields; the M2 diagnostic and
  M3 compilation artifacts do contain those fields.

The publication commit captures the source now present in the repository and
the verifier confirms that all published artifacts are mutually consistent. It
does not retroactively prove byte-for-byte identity of every untracked runner
at each historical execution time. Reports should preserve this distinction.
