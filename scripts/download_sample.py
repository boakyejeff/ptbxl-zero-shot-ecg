"""Download a balanced ~200-record sample from PTB-XL records500/00000/.

Reads the local metadata CSVs, picks a stratified sample (see src/meta.py),
then downloads <id>_hr.dat + <id>_hr.hea for each record with resume support.

Usage:  python scripts/download_sample.py [--n-per-class 40]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.meta import load_database, stratified_sample_ids  # noqa: E402

BASE = "https://physionet.org/files/ptb-xl/1.0.3/records500/00000/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rec_dir = data_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    db = load_database(data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv")
    sample = stratified_sample_ids(db, n_per_class=args.n_per_class, seed=args.seed)
    print(f"sampled {len(sample)} records")

    sample[["ecg_id", "superclass", "filename_hr"]].to_csv(
        data_dir / "sample_manifest.csv", index=False)

    ok, failed = 0, []
    for _, row in sample.iterrows():
        name = Path(row["filename_hr"]).name  # e.g. 00001_hr
        for ext in (".dat", ".hea"):
            url = BASE + name + ext
            dest = rec_dir / (name + ext)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            r = subprocess.run(
                ["curl", "-sS", "-C", "-", "--retry", "3", "--retry-delay", "2",
                 "--max-time", "300", "-o", str(dest), url],
                capture_output=True, text=True)
            if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
                failed.append(url)
                print(f"  FAIL {url}: {r.stderr.strip()[:120]}")
        ok += 1
        if ok % 25 == 0:
            print(f"  ... {ok}/{len(sample)} records")
    print(f"done: {ok} records attempted, {len(failed)} file failures")
    (data_dir / "download_failures.txt").write_text("\n".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
