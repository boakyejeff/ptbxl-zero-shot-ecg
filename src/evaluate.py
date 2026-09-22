"""Leave-one-superclass-out (LOSO) evaluation protocol.

For each held-out superclass c in {NORM, MI, STTC, CD, HYP}:
  1. Train the embedding model on the other 4 superclasses (4-way classifier).
  2. OOD detection: AUROC of ID (test samples of known classes) vs OOD
     (held-out-class samples), scored by max-softmax and by energy.
  3. 5-shot kNN: 5 labeled support examples of the held-out class + 5 per
     known class; classify held-out queries by nearest centroid in embedding
     space; report accuracy on the held-out-class queries.

Baselines repeat the same protocol on raw downsampled-signal features with a
plain nearest-centroid classifier (no learned embedding).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from .meta import SUPERCLASSES
from .model import EmbeddingMLP


@dataclass
class FoldResult:
    held_out: str
    n_train: int
    n_id_test: int
    n_ood: int
    ood_auroc_maxsoftmax: float
    ood_auroc_energy: float
    fewshot_acc_heldout: float
    fewshot_acc_overall: float
    baseline_ood_auroc: float
    baseline_fewshot_acc_heldout: float


def _train_test_split_known(Xk, yk, groups, seed):
    """Split the 4 known classes into train/test, grouped by patient if possible."""
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(Xk, yk, groups))
        # sanity: every class present on both sides
        if len(np.unique(yk[tr])) < len(np.unique(yk)) or len(np.unique(yk[te])) < len(
            np.unique(yk)
        ):
            raise ValueError("group split dropped a class")
        return tr, te
    except Exception:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(sss.split(Xk, yk))
        return tr, te


def _nearest_centroid_acc(E_sup, y_sup, E_qry, y_qry):
    clf = NearestCentroid()
    clf.fit(E_sup, y_sup)
    return float(clf.score(E_qry, y_qry))


def evaluate_fold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    held_out: str,
    n_support: int = 5,
    seed: int = 0,
    hidden: int = 96,
) -> FoldResult:
    rng = np.random.default_rng(seed)
    classes = np.array(SUPERCLASSES)

    m_known = y != held_out
    Xk, yk, gk = X[m_known], y[m_known], groups[m_known]
    Xo, yo = X[~m_known], y[~m_known]
    tr, te = _train_test_split_known(Xk, yk, gk, seed)

    # ---- learned model ----
    model = EmbeddingMLP(hidden=hidden, seed=seed).fit(Xk[tr], yk[tr])
    E_tr = model.embedding(Xk[tr])
    E_te = model.embedding(Xk[te])
    E_o = model.embedding(Xo)

    # (i) OOD detection: ID = known-class test, OOD = held-out class
    s_id = model.max_softmax(Xk[te])
    s_ood = model.max_softmax(Xo)
    y_true = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]
    scores = np.r_[s_id, s_ood]  # higher = more ID-like
    auroc_ms = float(roc_auc_score(y_true, scores))
    e_id = -model.energy(Xk[te])  # higher = more ID-like
    e_ood = -model.energy(Xo)
    auroc_en = float(roc_auc_score(y_true, np.r_[e_id, e_ood]))

    # (ii) 5-shot kNN in embedding space
    sup_idx, sup_lab = [], []
    # 5 support per known class from train
    for c in classes[classes != held_out]:
        idx = np.where(yk[tr] == c)[0]
        take = rng.choice(idx, size=min(n_support, len(idx)), replace=False)
        sup_idx += (tr[take]).tolist()
        sup_lab += [c] * len(take)
    # 5 support of the held-out class
    o_idx = np.arange(len(Xo))
    take_o = rng.choice(o_idx, size=min(n_support, len(o_idx)), replace=False)
    qry_mask = np.ones(len(Xo), dtype=bool)
    qry_mask[take_o] = False
    # map global train indices back to positions inside the train block
    tr_pos = {g: p for p, g in enumerate(tr)}
    E_sup = np.vstack([E_tr[[tr_pos[i] for i in sup_idx]], E_o[take_o]])
    y_sup = np.array(sup_lab + [held_out] * len(take_o))
    E_qry = np.vstack([E_o[qry_mask], E_te])
    y_qry = np.r_[np.full(qry_mask.sum(), held_out), yk[te]]
    fs_acc_overall = _nearest_centroid_acc(E_sup, y_sup, E_qry, y_qry)
    fs_acc_held = _nearest_centroid_acc(E_sup, y_sup, E_o[qry_mask], np.full(qry_mask.sum(), held_out))

    # ---- baseline: raw-signal features + plain nearest centroid ----
    # (X here is the engineered feature matrix; baseline uses its own raw matrix
    # passed via evaluate_loso's X_raw. We compute baseline inside evaluate_loso
    # and stash placeholder values here.)
    return FoldResult(
        held_out=held_out,
        n_train=len(tr),
        n_id_test=len(te),
        n_ood=len(Xo),
        ood_auroc_maxsoftmax=auroc_ms,
        ood_auroc_energy=auroc_en,
        fewshot_acc_heldout=fs_acc_held,
        fewshot_acc_overall=fs_acc_overall,
        baseline_ood_auroc=float("nan"),
        baseline_fewshot_acc_heldout=float("nan"),
    )


def evaluate_baseline_fold(
    Xr: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    held_out: str,
    n_support: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Same protocol on raw downsampled features with plain nearest-centroid.

    Returns (ood_auroc, fewshot_acc_on_heldout).
    """
    rng = np.random.default_rng(seed + 999)
    classes = np.array(SUPERCLASSES)
    m_known = y != held_out
    Xk, yk, gk = Xr[m_known], y[m_known], groups[m_known]
    Xo = Xr[~m_known]
    tr, te = _train_test_split_known(Xk, yk, gk, seed)

    scaler = StandardScaler().fit(Xk[tr])
    Zk_tr, Zk_te, Zo = scaler.transform(Xk[tr]), scaler.transform(Xk[te]), scaler.transform(Xo)

    # OOD: distance to nearest known centroid; ID should be closer.
    clf = NearestCentroid().fit(Zk_tr, yk[tr])
    d_id = pairwise_distances(Zk_te, clf.centroids_).min(axis=1)
    d_ood = pairwise_distances(Zo, clf.centroids_).min(axis=1)
    y_true = np.r_[np.ones(len(d_id)), np.zeros(len(d_ood))]
    auroc = float(roc_auc_score(y_true, np.r_[-d_id, -d_ood]))  # smaller dist = more ID

    # 5-shot nearest centroid on raw features
    sup_idx, sup_lab = [], []
    for c in classes[classes != held_out]:
        idx = np.where(yk[tr] == c)[0]
        take = rng.choice(idx, size=min(n_support, len(idx)), replace=False)
        sup_idx += tr[take].tolist()
        sup_lab += [c] * len(take)
    o_idx = np.arange(len(Xo))
    take_o = rng.choice(o_idx, size=min(n_support, len(o_idx)), replace=False)
    qry_mask = np.ones(len(Xo), dtype=bool)
    qry_mask[take_o] = False
    Z_sup = np.vstack([Zk_tr[np.array([list(tr).index(i) for i in sup_idx])], Zo[take_o]])
    y_sup = np.array(sup_lab + [held_out] * len(take_o))
    acc = _nearest_centroid_acc(Z_sup, y_sup, Zo[qry_mask], np.full(qry_mask.sum(), held_out))
    return auroc, float(acc)


def evaluate_loso(
    X: np.ndarray,
    X_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_support: int = 5,
    seed: int = 0,
    hidden: int = 96,
) -> list[FoldResult]:
    results = []
    for c in SUPERCLASSES:
        fr = evaluate_fold(X, y, groups, c, n_support=n_support, seed=seed, hidden=hidden)
        b_auroc, b_acc = evaluate_baseline_fold(X_raw, y, groups, c, n_support=n_support, seed=seed)
        fr.baseline_ood_auroc = b_auroc
        fr.baseline_fewshot_acc_heldout = b_acc
        results.append(fr)
    return results


def summarize(results: list[FoldResult]) -> dict:
    rows = [asdict(r) for r in results]
    keys = [k for k in rows[0] if k not in ("held_out",)]
    return {
        k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1))}
        for k in keys
    }
