"""Train or evaluate the nine ActProbe cells with a fixed, portable configuration."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"configs/table1.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(manifest):
    for rel, expected in manifest.items():
        path = ROOT/rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {rel}. Use a complete repository checkout.")
        if digest(path) != expected:
            raise ValueError(f"Release checksum mismatch: {rel}")


def worker(args, config):
    # Each benchmark exposes a package named `lib`, so isolate them in processes.
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for key in ["ACTPROBE_Q_AUC_MODE", "GROOT_ROOT", "PI0_ROOT", "PI05_ROOT", "OPENVLA_LIBERO_ROOT"]:
        os.environ.pop(key, None)
    benchmark = args.worker
    setting = config["benchmarks"][benchmark]
    sys.path[:0] = [str(ROOT/"benchmarks"/benchmark/"code"), str(ROOT)]
    import numpy as np
    import torch
    from lib import data, splits
    from lib.methods import actprobe
    from lib_shared import metrics
    from _table1_core import dump, loader, predict, score, train

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_files(setting["data_sha256"])
    verify_files({r["partition"]:r["partition_sha256"] for r in setting["runs"]
                  if not args.splits or r["split"] in args.splits})
    if args.mode == "evaluate":
        verify_files({r["checkpoint"]:r["checkpoint_sha256"] for r in setting["runs"]
                      if not args.splits or r["split"] in args.splits})
    episodes = data.load_metrics_logs()
    for task, items in episodes.items():
        for index, ep in enumerate(items):
            ep["audit_id"] = f"{task}/loader_row_{index}"
    if benchmark == "openvla_libero":
        episodes = data.subsample_50eps(episodes, seed=42)
    if {k:len(v) for k,v in episodes.items()} != setting["cohort_per_task"]:
        raise ValueError("Loaded cohort differs from the configured release")
    embeddings = data.load_qwen3_embeddings() if benchmark.startswith("pi05_") else data.load_task_embeddings()
    episodes = actprobe.subset_2feat(episodes)
    for task, items in episodes.items():
        for ep in items:
            key = ep["instruction"] if benchmark.startswith("pi05_") else task
            if ep["length"] != len(ep["raw"]) or ep["raw"].shape[1] != 2:
                raise ValueError(f"Bad feature dimensions: {ep['audit_id']}")
            if not np.isfinite(ep["raw"]).all() or key not in embeddings:
                raise ValueError(f"Bad features or missing embedding: {ep['audit_id']}")
            if np.asarray(embeddings[key]).shape != (1024,) or not np.isfinite(embeddings[key]).all():
                raise ValueError(f"Bad embedding: {key}")
    selected = [r for r in setting["runs"] if not args.splits or r["split"] in args.splits]
    if not selected:
        raise ValueError(f"No configured cells selected for {benchmark}")
    rows = []
    for run in selected:
        split, run_id = run["split"], run["id"]
        p = args.output/benchmark/f"{split}_{run_id}"
        p.mkdir(parents=True, exist_ok=False)
        fixed = json.loads((ROOT/run["partition"]).read_text())
        if set(fixed) != {"train", "val", "test"}:
            raise ValueError(f"Invalid partition groups for {benchmark}/{split}/{run_id}")
        episode_index = {e["audit_id"]:e for items in episodes.values() for e in items}
        if len(episode_index) != sum(map(len, episodes.values())):
            raise ValueError("Duplicate trajectory identifiers in evaluated cohort")
        unknown = set().union(*(set(fixed[g]) for g in fixed)) - set(episode_index)
        if unknown:
            raise ValueError(f"Partition contains unknown trajectory identifiers: {len(unknown)}")
        partition = {g:[episode_index[i] for i in fixed[g]] for g in ["train","val","test"]}
        ids = {g:[e["audit_id"] for e in partition[g]] for g in ["train","val","test"]}
        sets = [set(ids[g]) for g in ids]
        if any(len(sets[i]) != len(ids[g]) for i,g in enumerate(ids)) or \
                any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Duplicate or overlapping trajectory partitions")
        if split == "unseen" and ({e["task_id"] for e in partition["test"]} &
                {e["task_id"] for e in partition["train"]+partition["val"]}):
            raise ValueError("Unseen-task partition overlaps training or validation tasks")
        if set().union(*sets) != set(episode_index):
            raise ValueError("Fixed partition does not cover the evaluated cohort")
        dump(p/"partitions.json", ids)
        if args.mode == "train":
            model, mean, std, epoch = train(partition, embeddings, splits, actprobe,
                benchmark, p, args.epochs)
            checkpoint = p/"model.pt"
            torch.save(dict(model_state_dict=model.state_dict(), norm_mean=mean.tolist(),
                norm_std=std.tolist(), selected_epoch=epoch,
                feat_idx=list(actprobe.PAPER_FEAT_IDX)), checkpoint)
        else:
            checkpoint = ROOT/run["checkpoint"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = actprobe.ActProbeNet()
            model.load_state_dict(saved["model_state_dict"])
            mean = np.array(saved["norm_mean"], dtype=np.float32)
            std = np.array(saved["norm_std"], dtype=np.float32)
            epoch = saved.get("selected_epoch")
        if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or \
                not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid checkpoint normalization")
        dump(p/"locked_checkpoint.json", dict(sha256=digest(checkpoint), selected_epoch=epoch,
            run=run_id, partition_sha256=run["partition_sha256"]))
        predictions = predict(model, loader(partition["test"], embeddings, mean, std, benchmark))
        auc, values, cutoffs = score(predictions)
        if abs(auc - 100*metrics.q_auc(predictions, mode="taskmax", q=.25)) > 1e-10:
            raise ValueError("Independent AUROC disagrees with shared metric")
        reference = actprobe.score_episodes(model, partition["test"][:3], embeddings, mean, std, device="cpu")
        for actual, expected in zip(predictions[:3], reference):
            np.testing.assert_allclose(actual["scores"], expected["scores"], rtol=2e-5, atol=2e-6)
        np.savez_compressed(p/"predictions.npz", **{f"episode_{j}":r["scores"] for j,r in enumerate(predictions)})
        dump(p/"episodes.json", [dict(audit_id=e["audit_id"], task_id=e["task_id"],
            label=int(e["label"]), length=e["length"], cutoff=cut, score=value)
            for e,cut,value in zip(partition["test"], cutoffs, values)])
        row = dict(benchmark=benchmark, split=split, run=run_id, auc=auc,
            checkpoint_sha256=digest(checkpoint), counts={g:len(ids[g]) for g in ids})
        if args.mode == "evaluate":
            row["reference_auc"] = run["reference_auc"]
            if abs(auc-run["reference_auc"]) > .01:
                raise ValueError(f"Reference AUC mismatch for {benchmark}/{split}/{run_id}: {auc}")
        dump(p/"result.json", row)
        rows.append(row)
        print(f"{benchmark} {split} {run_id}: {auc:.4f}", flush=True)
    dump(args.output/benchmark/"results.json", dict(rows=rows, versions=dict(
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate","train"], default="evaluate")
    parser.add_argument("--benchmarks", nargs="+", help="Default: all five benchmarks / nine cells")
    parser.add_argument("--splits", nargs="+", choices=["allseen","unseen"])
    parser.add_argument("--output", type=Path, help="A new directory (existing output is never reused)")
    parser.add_argument("--epochs", type=int, help="Short training budget for smoke checks; not reference results")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.epochs is not None and args.mode != "train":
        parser.error("--epochs only applies to --mode train")
    if args.worker:
        worker(args, config)
        return
    selected = args.benchmarks or list(config["benchmarks"])
    if not selected or len(set(selected)) != len(selected) or any(b not in config["benchmarks"] for b in selected):
        parser.error("Choose distinct benchmarks from: " + ", ".join(config["benchmarks"]))
    for b in selected:
        if args.splits and not any(r["split"] in args.splits for r in config["benchmarks"][b]["runs"]):
            parser.error(f"No requested split is available for {b}")
    # Verify the complete requested input set before any potentially long work.
    for b in selected:
        verify_files({r["partition"]:r["partition_sha256"] for r in config["benchmarks"][b]["runs"]
                      if not args.splits or r["split"] in args.splits})
        verify_files(config["benchmarks"][b]["data_sha256"])
        if args.mode == "evaluate":
            verify_files({r["checkpoint"]:r["checkpoint_sha256"] for r in config["benchmarks"][b]["runs"]
                          if not args.splits or r["split"] in args.splits})
    output = args.output or ROOT/"results"/f"table1_{args.mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    args.output = output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from _table1_core import dump
    dump(args.output/"run.json", dict(mode=args.mode, benchmarks=selected, splits=args.splits,
        epochs=args.epochs, config_sha256=digest(CONFIG), device="cpu", metric=config["metric"],
        command=sys.argv, source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [
            Path(__file__), ROOT/"scripts/_table1_core.py", ROOT/"lib_shared/metrics.py"]}))
    for benchmark in selected:
        cmd = [sys.executable, str(Path(__file__)), "--worker", benchmark, "--mode", args.mode,
               "--output", str(args.output)]
        if args.splits:
            cmd += ["--splits", *args.splits]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        subprocess.run(cmd, check=True, cwd=ROOT)
    import numpy as np
    rows = []
    for benchmark in selected:
        results = json.loads((args.output/benchmark/"results.json").read_text())["rows"]
        for split in ["allseen","unseen"]:
            members = [r for r in results if r["split"] == split]
            if not members:
                continue
            values = [r["auc"] for r in members]
            rows.append(dict(benchmark=benchmark, split=split,
                runs=[r["run"] for r in members],
                mean=float(np.mean(values)), std_population=float(np.std(values)),
                paper_target=config["benchmarks"][benchmark]["paper_targets"][split]))
    dump(args.output/"summary.json", dict(mode=args.mode, training_epochs_limit=args.epochs,
        metric=config["metric"], rows=rows))
    with (args.output/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nBenchmark / split                         AUROC mean ± std")
    for row in rows:
        print(f"{row['benchmark']+' / '+row['split']:<42} {row['mean']:.2f} ± {row['std_population']:.2f}")
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, ModuleNotFoundError,
            subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
