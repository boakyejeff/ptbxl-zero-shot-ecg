"""Metadata loading and dominant-superclass mapping for PTB-XL."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_code_to_superclass(scp_statements_csv: str | Path) -> dict[str, str]:
    """Map every diagnostic SCP code to one of the 5 diagnostic superclasses."""
    scp = pd.read_csv(scp_statements_csv, index_col=0)
    diag = scp[scp["diagnostic"] == 1.0]
    mapping = {}
    for code, row in diag.iterrows():
        s = row["diagnostic_class"]
        if s in SUPERCLASSES:
            mapping[str(code)] = s
    return mapping


def dominant_superclass(scp_codes: str, code2super: dict[str, str]) -> str | None:
    """Aggregate likelihoods per superclass and take argmax (standard PTB-XL recipe)."""
    try:
        codes = ast.literal_eval(scp_codes)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(codes, dict):
        return None
    agg: dict[str, float] = {}
    for code, likelihood in codes.items():
        s = code2super.get(str(code))
        if s is not None:
            agg[s] = agg.get(s, 0.0) + float(likelihood)
    if not agg:
        return None
    return max(agg, key=agg.get)


def load_database(
    ptbxl_database_csv: str | Path,
    scp_statements_csv: str | Path,
) -> pd.DataFrame:
    """Load PTB-XL metadata and attach a dominant-superclass label per record.

    Records with no diagnostic superclass are dropped.
    """
    code2super = load_code_to_superclass(scp_statements_csv)
    db = pd.read_csv(ptbxl_database_csv)
    db["superclass"] = db["scp_codes"].apply(lambda d: dominant_superclass(d, code2super))
    db = db[db["superclass"].notna()].copy()
    return db


def stratified_sample_ids(
    db: pd.DataFrame,
    dir_prefix: str = "records500/00000/",
    n_per_class: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample a balanced subset of records available under dir_prefix.

    Filters to filenames inside dir_prefix (e.g. records500/00000/), then takes
    up to n_per_class records per superclass, preferring distinct patients.
    """
    sub = db[db["filename_hr"].str.startswith(dir_prefix)].copy()
    rng = np.random.default_rng(seed)
    picks = []
    for s in SUPERCLASSES:
        cand = sub[sub["superclass"] == s]
        if len(cand) == 0:
            continue
        # Prefer distinct patients to reduce within-patient leakage
        cand = cand.sort_values("patient_id").drop_duplicates("patient_id", keep="first")
        take = min(n_per_class, len(cand))
        idx = rng.choice(cand.index.to_numpy(), size=take, replace=False)
        picks.append(cand.loc[idx])
    out = pd.concat(picks).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out
