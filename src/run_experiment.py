"""Run the full LOSO experiment.

Usage:
    python -m src.run_experiment --data-dir data --out results --n-per-class 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import load_signals
from .evaluate import evaluate_loso, summarize
from .features import feature_matrix, raw_downsampled_features
from .meta import load_database, stratified_sample_ids

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--n-support", type=int, default=5)
    ap.add_argument("--cache", default="features_cache.npz",
                    help="npz cache of engineered features in --out")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading metadata ...", flush=True)
    db = load_database(data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv")
    sample = stratified_sample_ids(db, n_per_class=args.n_per_class, seed=args.seed)
    print(f"      sampled {len(sample)} records: "
          + ", ".join(f"{c}={int((sample['superclass'] == c).sum())}"
                      for c in ["NORM", "MI", "STTC", "CD", "HYP"]), flush=True)

    print("[2/5] reading waveforms ...", flush=True)
    signals = load_signals(sample, data_dir)
    order = sample["ecg_id"].astype(int).tolist()
    y = sample["superclass"].to_numpy()
    groups = sample["patient_id"].to_numpy()

    cache_path = out_dir / args.cache
    if cache_path.exists():
        print("[3/5] loading cached features ...", flush=True)
        z = np.load(cache_path)
        X, X_raw = z["X"], z["X_raw"]
        assert X.shape[0] == len(order), "cache/order mismatch; delete the cache"
    else:
        print("[3/5] extracting features ...", flush=True)
        X = feature_matrix(signals, order)
        X_raw = np.stack([raw_downsampled_features(signals[e]) for e in order])
        np.savez_compressed(cache_path, X=X, X_raw=X_raw)
    print(f"      engineered: {X.shape}, raw: {X_raw.shape}", flush=True)

    print("[4/5] LOSO evaluation (5 folds) ...", flush=True)
    results = evaluate_loso(X, X_raw, y, groups,
                            n_support=args.n_support, seed=args.seed, hidden=args.hidden)

    print("[5/5] writing results ...", flush=True)
    rows = []
    for r in results:
        rows.append({
            "held_out": r.held_out, "n_train": r.n_train,
            "n_id_test": r.n_id_test, "n_ood": r.n_ood,
            "ood_auroc_maxsoftmax": round(r.ood_auroc_maxsoftmax, 4),
            "ood_auroc_energy": round(r.ood_auroc_energy, 4),
            "fewshot_acc_heldout": round(r.fewshot_acc_heldout, 4),
            "fewshot_acc_overall": round(r.fewshot_acc_overall, 4),
            "baseline_ood_auroc": round(r.baseline_ood_auroc, 4),
            "baseline_fewshot_acc_heldout": round(r.baseline_fewshot_acc_heldout, 4),
        })
    pd.DataFrame(rows).to_csv(out_dir / "loso_results.csv", index=False)
    summ = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(
        {"per_fold": rows, "mean_std": summ}, indent=2))

    hdr = ["held_out", "OOD-AUROC(ms)", "OOD-AUROC(en)", "5shot-acc(held)",
           "5shot-acc(all)", "base-AUROC", "base-5shot"]
    print("\n" + " | ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{v:>14}" for v in [
            r["held_out"], r["ood_auroc_maxsoftmax"], r["ood_auroc_energy"],
            r["fewshot_acc_heldout"], r["fewshot_acc_overall"],
            r["baseline_ood_auroc"], r["baseline_fewshot_acc_heldout"]]))
    print("\nmean ± std over 5 folds:")
    for k, v in summ.items():
        print(f"  {k:32s} {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"\nwrote {out_dir / 'loso_results.csv'} and {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
