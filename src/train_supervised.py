"""Supervised 1D-ResNet classifier for PTB-XL diagnostic superclasses.

Trains on the locally downloaded _hr (100 Hz) record set using
patient-grouped 70/15/15 train/val/test splits (no patient in two splits),
stratified by each patient's majority diagnostic superclass.
Labels: dominant diagnostic superclass (NORM, MI, STTC, CD, HYP).

CPU-friendly by design (~974k params, streams records from disk, batch 64).

Usage:
    <venv-python> src/train_supervised.py [--epochs 30] [--batch 64]

Writes: results/supervised_metrics.json, results/supervised_confusion.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.dataio import local_record_path, read_record  # noqa: E402
from src.meta import SUPERCLASSES, load_database  # noqa: E402

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

LABEL2IDX = {s: i for i, s in enumerate(SUPERCLASSES)}
torch.set_num_threads(2)


# ---------------------------------------------------------------- data

class ECGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.augment = augment
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        sig = read_record(local_record_path(self.data_dir, row["filename_hr"]))  # (1000, 12)
        x = sig.T.copy()  # (12, 1000)
        # per-lead z-score
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        x = (x - mu) / sd
        if self.augment:
            shift = int(self.rng.integers(-100, 101))
            x = np.roll(x, shift, axis=1)
            x = x + self.rng.normal(0, 0.01, x.shape).astype(np.float32)
        return (torch.from_numpy(x.astype(np.float32)),
                torch.tensor(LABEL2IDX[row["superclass"]], dtype=torch.long))


# ---------------------------------------------------------------- model

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out))

    def forward(self, x):
        r = x if self.down is None else self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + r)


class ECGResNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))          # (64, 250)
        self.b1 = ResBlock(64, 64)                          # (64, 250)
        self.b2 = ResBlock(64, 128, stride=2)               # (128, 125)
        self.b3 = ResBlock(128, 256, stride=2)              # (256, 63)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b3(self.b2(self.b1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- train

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            ys.append(y.numpy())
            ps.append(torch.softmax(logits, 1).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    per_class_auroc = {}
    for i, s in enumerate(SUPERCLASSES):
        try:
            per_class_auroc[s] = float(roc_auc_score((y == i).astype(int), p[:, i]))
        except ValueError:
            per_class_auroc[s] = float("nan")
    return {
        "macro_auroc": float(np.nanmean(list(per_class_auroc.values()))),
        "per_class_auroc": per_class_auroc,
        "accuracy": float(accuracy_score(y, p.argmax(1))),
        "macro_f1": float(f1_score(y, p.argmax(1), average="macro")),
        "y_true": y, "y_prob": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    data_dir = Path(args.data_dir)
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    db = load_database(data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv")

    # Keep only records actually downloaded locally (records/ is a partial set).
    has_file = db["filename_hr"].apply(
        lambda f: local_record_path(data_dir, f).with_suffix(".dat").exists())
    db = db[has_file].copy()
    print(f"records with local files: {len(db)}", flush=True)

    # Patient-grouped train/val/test split (70/15/15), stratified by each
    # patient's majority superclass — no patient appears in two splits.
    rng = np.random.default_rng(args.seed)
    pat = (db.groupby("patient_id")["superclass"]
             .agg(lambda s: s.mode().iloc[0]).reset_index())
    tr_ids, va_ids, te_ids = [], [], []
    for s in SUPERCLASSES:
        ids = pat.loc[pat["superclass"] == s, "patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_te, n_va = int(round(n * 0.15)), int(round(n * 0.15))
        te_ids += ids[:n_te].tolist()
        va_ids += ids[n_te:n_te + n_va].tolist()
        tr_ids += ids[n_te + n_va:].tolist()
    tr = db[db["patient_id"].isin(tr_ids)].copy()
    va = db[db["patient_id"].isin(va_ids)].copy()
    te = db[db["patient_id"].isin(te_ids)].copy()
    assert not (set(tr["patient_id"]) & set(va["patient_id"]) & set(te["patient_id"]))
    print(f"train={len(tr)} val={len(va)} test={len(te)} "
          f"(patient-grouped 70/15/15, seed {args.seed})", flush=True)
    print("train class counts:", tr["superclass"].value_counts().to_dict(), flush=True)

    counts = tr["superclass"].value_counts()
    w = torch.tensor([len(tr) / (len(SUPERCLASSES) * counts[s]) for s in SUPERCLASSES],
                     dtype=torch.float32)

    train_loader = DataLoader(ECGDataset(tr, data_dir, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(ECGDataset(va, data_dir), batch_size=args.batch, num_workers=0)
    test_loader = DataLoader(ECGDataset(te, data_dir), batch_size=args.batch, num_workers=0)

    model = ECGResNet().to(device)
    print(f"params: {count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3)

    best_va, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        vm = evaluate(model, val_loader, device)
        sched.step(vm["macro_auroc"])
        el = time.time() - t0
        print(f"epoch {epoch+1:02d} train_loss={tot/n:.4f} "
              f"val_macroAUROC={vm['macro_auroc']:.4f} val_acc={vm['accuracy']:.4f} "
              f"({el/60:.1f}m)", flush=True)
        if vm["macro_auroc"] > best_va + 1e-4:
            best_va, bad = vm["macro_auroc"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    tm = evaluate(model, test_loader, device)
    cm = confusion_matrix(tm["y_true"], tm["y_prob"].argmax(1)).tolist()

    metrics = {
        "model": "ECGResNet-1D (~974k params)",
        "data": "PTB-XL v1.0.3 _hr (100 Hz), dominant diagnostic superclass labels",
        "splits": {"train": "patient-grouped 70%", "val": "patient-grouped 15%",
                   "test": "patient-grouped 15% (no patient in two splits)",
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
        "best_val_macro_auroc": best_va,
        "test_macro_auroc": tm["macro_auroc"],
        "test_per_class_auroc": tm["per_class_auroc"],
        "test_accuracy": tm["accuracy"],
        "test_macro_f1": tm["macro_f1"],
        "test_confusion_matrix": {"labels": SUPERCLASSES, "matrix": cm},
        "seed": args.seed,
    }
    with open(res_dir / "supervised_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(res_dir / "supervised_confusion.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["true/pred"] + SUPERCLASSES)
        for s, row in zip(SUPERCLASSES, cm):
            wr.writerow([s] + row)

    print("\n=== TEST (patient-grouped held-out, 15%) ===", flush=True)
    for s in SUPERCLASSES:
        print(f"  {s:5s} AUROC {tm['per_class_auroc'][s]:.4f}", flush=True)
    print(f"  macro AUROC {tm['macro_auroc']:.4f} | acc {tm['accuracy']:.4f} | "
          f"macro F1 {tm['macro_f1']:.4f}", flush=True)
    print(f"saved to {res_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
