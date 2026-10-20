# ActProbe: Action-Space Probe for Early Failure Detection of Generative Robot Policies

[Project Page](https://air-embodied-brain.github.io/actprobe/) &middot; [Paper (arXiv)](https://arxiv.org/abs/2606.08508) &middot; Preprint

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="ActProbe raises an alarm before failures become visually recognizable.">
</p>

<p align="center"><em>ActProbe raises an alarm before failures become visually recognizable by probing action-space signals at runtime; the strongest hidden-state baseline only fires once the failure has already happened.</em></p>

**ActProbe** is a lightweight, *pure action-space* detector that anticipates failures of generative
robot policies (VLAs and world-action models) before they become visually recognizable. It reads two
compact signals directly from the emitted action chunks — **Temporal Consistency Error (TCE)** between
consecutive chunks and **Action Chunk Magnitude (ACM)** of the current chunk — and maps them to
per-step failure probabilities with a small (~24K-parameter) task-conditioned LSTM–MLP. No white-box
access, no resampling, no observation-side fusion.

This repository contains the benchmark code, the training data, and the scripts to train and evaluate
ActProbe across five policy–environment settings.

## Installation

For the one-command checkpoint evaluation below, start with Python 3.10 or 3.11.
The launcher creates an isolated CPU environment on its first run. No GPU, VLA
installation, rollout collection, or embedding-model download is needed.

Clone the repository:

```bash
git clone https://github.com/air-embodied-brain/actprobe.git
cd actprobe
```

Run all training and evaluation commands below from the repository root.
The one-command launcher installs its own dependencies; no `source env.sh` step is needed.
For other benchmark-specific scripts, install the general dependencies with
`pip install -r requirements.txt` and follow their environment setup.

## One-command Table 1 ActProbe evaluation

```bash
bash scripts/run_table1.sh
```

This evaluates the bundled ActProbe checkpoints for all **nine Table 1 settings**.
The first run installs the CPU dependencies; subsequent runs reuse the environment.
Input data and checkpoints are checked against their SHA-256 hashes before evaluation.
Results are written to a new `results/table1_evaluate_<timestamp>/` directory:
`summary.json`, `summary.csv`, per-run partitions, and per-trajectory predictions.

Useful options:

```bash
# Evaluate just RoboCasaL.
bash scripts/run_table1.sh --benchmarks pi05_robocasa_multistage

# Use an existing compatible Python environment.
ACTPROBE_PYTHON=/path/to/python bash scripts/run_table1.sh

# Train new probes, then evaluate them, using the same configured partitions.
bash scripts/run_table1.sh --mode train --benchmarks pi05_robocasa_multistage

# Train and evaluate all nine settings (considerably slower than checkpoint evaluation).
bash scripts/run_table1.sh --mode train
```

Training writes fresh checkpoints under its output directory and never overwrites the
bundled weights. The baseline training recipe is implemented in `scripts/_table1_core.py`;
complete historical training recipes are not available for every archived checkpoint.
Consequently, fresh training and bundled-checkpoint evaluation are separate workflows.

## Data

This release is self-contained for **ActProbe**. Each benchmark under `benchmarks/<name>/` ships the
per-step action-space features used to train and evaluate the probe plus the task embeddings used for
task conditioning. The 27 small ActProbe checkpoints used by the one-command evaluator
are included under `artifacts/table1/`. The separate baseline detectors' weights are not included.

| Benchmark | Shipped ActProbe data | Task-conditioning data |
|---|---|---|
| `pi0_libero` | `data/metrics_logs/task_{0..9}.jsonl` | `data/task_embeddings.npy`, `data/task_instructions.json` |
| `openvla_libero` | `data/metrics_logs/task_{0..9}.jsonl` | `data/task_embeddings.npy`, `data/task_instructions.json` |
| `groot_robocasa` | `data/metrics_logs/*.jsonl` | `data/task_embeddings.npy`, `data/task_instructions.json` |
| `pi05_robocasa` | `data/probe_data/*.pkl` | `data/qwen3_emb.pkl` |
| `pi05_robocasa_multistage` | `data/probe_data/*.pkl` | `data/qwen3_emb.pkl` |

For exact field schemas and feature ordering, use each benchmark's loader as the source of truth:
`benchmarks/<name>/code/lib/data.py`.

Some baseline detectors require additional hidden-state data or pretrained weights that are
not bundled with this release. The one-command workflow evaluates ActProbe only.

## Train and evaluate ActProbe

Run from the repository root. For example, to train or evaluate GR00T:

```bash
# Train and evaluate using the released fixed partitions.
bash scripts/run_table1.sh --mode train --benchmarks groot_robocasa

# Evaluate the bundled checkpoints.
bash scripts/run_table1.sh --benchmarks groot_robocasa
```

Training outputs, including new checkpoints, are saved under
`results/table1_train_<timestamp>/`. Evaluation outputs are saved under
`results/table1_evaluate_<timestamp>/`.

## Reproducing the feature data (optional)

You do **not** need this to train or evaluate ActProbe — the shipped `data/` already contains the
per-step features. We include the collection code under `benchmarks/<name>/scripts/collect/` so the
features can be regenerated from raw policy rollouts. (ActProbe itself uses only two of them — **TCE**
(`chunk_mse`) and **ACM** (`action_norm`); the rest are kept for completeness and ablations.)

Collection takes one of two forms, depending on the policy:

- **Post-hoc extraction** — offline, on saved rollout records, no VLA required:
  - `pi0_libero/scripts/collect/extract_features.py` — reads π0 `env_records/` + `policy_records/`.
  - `openvla_libero/scripts/collect/extract_features.py` — reads OpenVLA per-episode CSV + PKL.
- **Online logging during rollout** (reference) — features are logged as the policy runs, because the
  denoising / ODE-trajectory curvature is produced by the policy itself. These need the corresponding
  VLA stack (GR00T + robosuite, or an openpi π0.5 server):
  - `groot_robocasa/scripts/collect/rollout_log_metrics.py`
  - `pi05_robocasa/scripts/collect/eval_log_metrics.py` — the multi-stage benchmark reuses this by
    passing a multi-stage task list via `--tasks`.

The split is a matter of provenance. The LIBERO rollout records consumed by the post-hoc extractors are
produced by the SAFE / [vla-safe](https://github.com/vla-safe/SAFE) rollout pipelines, which dump
per-step `env_records`/`policy_records` (and CSV logs) to disk — so we extract features from those dumps
offline. The RoboCasa benchmarks have no such pre-dumped archive: the policies are rolled out directly
and the (compact) features are logged online during the run.

The released LIBERO and GR00T feature data use JSONL files, with one episode per line:
`{episode_id, success, length, steps: [{t, chunk_mse, action_norm, ...}, ...]}`.
The two π0.5 benchmarks instead use per-task PKL files containing episode records.
See each benchmark's `code/lib/data.py` for its exact loading format.

## RL fine-tuning

ActProbe also accelerates online RL fine-tuning by early-terminating rollouts it predicts will fail.
The PPO integration for LIBERO (built on [RLinf](https://github.com/RLinf/RLinf)) lives in a separate
fork:

> **RL code:** https://github.com/BigBen111/rlinf-actprobe

See that repository's README for the launch configs and setup.

## Benchmarks

| Benchmark | Policy | Environment | Action type |
|---|---|---|---|
| `pi0_libero`               | π0       | LIBERO-10                  | chunk          |
| `openvla_libero`           | OpenVLA  | LIBERO-10                  | autoregressive |
| `groot_robocasa`           | GR00T    | RoboCasa (single-stage)    | chunk          |
| `pi05_robocasa`            | π0.5     | RoboCasa (single-stage)    | chunk          |
| `pi05_robocasa_multistage` | π0.5     | RoboCasa (multi-stage)     | chunk          |

Each benchmark directory contains:
- `code/` — method implementations (`lib/`), training (`train/`), and evaluation (`eval/`)
- `scripts/` — data preprocessing, entry points, and `collect/` (feature-collection code)
- `env.sh` — environment variables (paths, GPU, Python interpreter)

## Methods

| Method | Type | Reference |
|---|---|---|
| **ActProbe** (ours)              | learned: 2 action-space features + LSTM–MLP | this paper |
| SAFE-MLP / SAFE-LSTM             | learned: hidden-state probes                | Gu et al., NeurIPS 2025 |
| SAFE-MLP-TDQC / SAFE-LSTM-TDQC   | learned: temporal-difference calibrated     | Francis-Meretzki et al., 2026 |
| LogpZO                           | score-based density over action sequences   | Xu et al., 2025 |
| STAC-Single                      | chunk-overlap disagreement                  | Agia et al., CoRL 2024 |
| Cosine k-NN                      | non-parametric hidden-state baseline        | — |

## Citation

```bibtex
@misc{actprobe2026,
  title  = {ActProbe: Action-Space Probe for Early Failure Detection of Generative Robot Policies},
  author = {Huang, Bingjia and Li, Xiangyu and Wang, Xiang and Mi, Liang and Hao, Zixu and
            Wang, Weijun and Wu, Hao and Li, Kun and Liu, Yunxin and Cao, Ting},
  year   = {2026},
  eprint = {2606.08508},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)).
