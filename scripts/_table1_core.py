"""Deterministic CPU training and explicit taskmax scoring for the Table 1 runner."""
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CachedDataset(Dataset):
    def __init__(self, episodes, embeddings, mean, std, max_len, benchmark):
        self.items = []
        for ep in episodes:
            length = ep["length"]
            ts = (np.arange(length, dtype=np.float32) / 100.).reshape(-1, 1)
            x = np.hstack([(ep["raw"] - mean) / (std + 1e-7), ts]).astype(np.float32)
            x = np.pad(x, ((0, max_len-length), (0, 0)))
            key = ep["instruction"] if benchmark.startswith("pi05_") else ep["task_id"]
            self.items.append((torch.from_numpy(x),
                torch.from_numpy(embeddings[key].astype(np.float32)),
                torch.tensor(length), torch.tensor(float(ep["label"])), ep["task_id"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    x, lang, lens, labels, tasks = zip(*batch)
    return torch.stack(x), torch.stack(lang), torch.stack(lens), torch.stack(labels), list(tasks)


def loader(episodes, embeddings, mean, std, benchmark, shuffle=False, max_len=None):
    return DataLoader(CachedDataset(episodes, embeddings, mean, std,
        max_len or max(e["length"] for e in episodes), benchmark),
        batch_size=64, shuffle=shuffle, collate_fn=collate, num_workers=0)


@torch.no_grad()
def predict(model, batches):
    model.eval()
    result = []
    for x, lang, lengths, labels, tasks in batches:
        scores = model(x, lang, lengths)
        for i, (length, label, task) in enumerate(zip(lengths.tolist(), labels.tolist(), tasks)):
            result.append(dict(scores=scores[i, :length].numpy().copy(), length=length,
                               label=int(label), task_id=task))
    return result


def score(predictions):
    """Failure-positive AUROC; caps are computed within this evaluation set."""
    if not predictions or len({p["label"] for p in predictions}) != 2:
        raise ValueError("AUROC requires a nonempty evaluation set with both labels")
    maxima = {}
    for p in predictions:
        maxima[p["task_id"]] = max(maxima.get(p["task_id"], 0), p["length"])
    values, cutoffs = [], []
    for p in predictions:
        cut = min(p["length"], math.ceil(.25 * maxima[p["task_id"]]))
        if cut < 1 or not np.isfinite(p["scores"]).all():
            raise ValueError("Invalid episode length or non-finite prediction")
        values.append(float(p["scores"][:cut].max()))
        cutoffs.append(cut)
    return float(roc_auc_score([p["label"] for p in predictions], values)*100), values, cutoffs


def train(partition, embeddings, splits, model_module, benchmark, output, epochs=None):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    tr, va = partition["train"], partition["val"]
    mean, std = splits.fit_norm(tr, key="raw")
    max_len = max(e["length"] for e in tr + va)
    tr_loader = loader(tr, embeddings, mean, std, benchmark, True, max_len)
    va_loader = loader(va, embeddings, mean, std, benchmark, False, max_len)
    model = model_module.ActProbeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    maximum = 300 if benchmark == "groot_robocasa" else 400
    every = 10 if benchmark == "groot_robocasa" else 5
    if epochs is not None and (epochs < every or epochs > maximum):
        raise ValueError(f"--epochs must be between {every} and {maximum} for {benchmark}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum)
    best, best_state, best_epoch, stale = -1., None, 0, 0
    history = []
    start = time.monotonic()
    for epoch in range(1, (epochs or maximum)+1):
        model.train()
        for x, lang, lengths, labels, _ in tr_loader:
            scores = model(x, lang, lengths)
            mask = torch.arange(scores.shape[1])[None, :] < lengths[:, None]
            errors = torch.nn.functional.binary_cross_entropy(scores.clamp(1e-7, 1-1e-7),
                labels[:, None].expand_as(scores), reduction="none")
            loss = (errors * mask).sum(1).sum() / lengths.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        scheduler.step()
        if epoch % every == 0:
            auc = score(predict(model, va_loader))[0]/100
            if auc > best + 1e-4:
                best, best_epoch, best_state, stale = auc, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += every
            history.append(dict(epoch=epoch, validation_auc=auc*100, best_epoch=best_epoch))
            dump(output/"history.json", dict(history=history, finished=False))
            if benchmark != "groot_robocasa" and stale >= 50:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a validated checkpoint")
    model.load_state_dict(best_state)
    dump(output/"history.json", dict(history=history, finished=True, selected_epoch=best_epoch,
        epochs_run=epoch, elapsed_seconds=time.monotonic()-start))
    return model, mean, std, best_epoch
