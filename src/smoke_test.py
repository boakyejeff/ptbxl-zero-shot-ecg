"""Smoke test: end-to-end pipeline on a handful of downloaded records.

Runs in ~1-2 minutes. Usage:  python -m src.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    from .dataio import load_signals
    from .features import extract_features, feature_matrix
    from .meta import load_database

    data_dir = REPO / "data"
    assert (data_dir / "ptbxl_database.csv").exists(), "metadata CSV missing"
    recs = sorted((data_dir / "records").glob("*_hr.hea"))
    assert len(recs) >= 6, f"need >=6 records under data/records, found {len(recs)}"
    print(f"[ok] metadata present; {len(recs)} records downloaded")

    db = load_database(data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv")
    # take up to 12 NORM + 12 MI records that we actually downloaded
    have = {int(p.name.split("_")[0]) for p in recs}
    cand = db[db["ecg_id"].astype(int).isin(have)]
    pick = pd_concat_pick(cand, per_class=12)
    assert len(pick) >= 20, f"only {len(pick)} labeled records available"
    signals = load_signals(pick, data_dir)
    for eid, sig in signals.items():
        assert sig.shape == (5000, 12) and sig.dtype == np.float32
    print(f"[ok] loaded {len(signals)} waveforms, all (5000, 12) float32")

    X = feature_matrix(signals, pick["ecg_id"].astype(int).tolist())
    assert X.shape == (len(pick), 240) and np.isfinite(X).all()
    print(f"[ok] features {X.shape}, finite")

    # tiny 2-way sanity fit
    from sklearn.preprocessing import LabelEncoder
    from .model import EmbeddingMLP
    y = LabelEncoder().fit_transform(pick["superclass"].to_numpy())
    m = EmbeddingMLP(hidden=16, max_iter=200, seed=0).fit(X, y)
    acc = m.mlp.score(m.scaler.transform(X), y)
    assert acc > 0.5, f"suspicious train acc {acc}"
    E = m.embedding(X)
    assert E.shape == (len(pick), 16)
    print(f"[ok] MLP trains (train acc {acc:.2f}); embeddings {E.shape}")
    print("SMOKE TEST PASSED")
    return 0


def pd_concat_pick(cand, per_class=12):
    import pandas as pd
    frames = []
    for s in ("NORM", "MI"):
        frames.append(cand[cand["superclass"] == s].head(per_class))
    return pd.concat(frames).reset_index(drop=True)


if __name__ == "__main__":
    sys.exit(main())
