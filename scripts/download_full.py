"""Download the full PTB-XL v1.0.3 _hr (100 Hz) record set from PhysioNet.

Reads the local metadata CSVs, keeps every record that maps to one of the 5
diagnostic superclasses (~21.8k records), and downloads <id>_hr.dat +
<id>_hr.hea in parallel with retries and resume-skip of existing files.

Usage:  python scripts/download_full.py [--jobs 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.meta import load_database  # noqa: E402

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
lock = threading.Lock()
stats = {"ok": 0, "skipped": 0, "failed": []}


def fetch(url: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        with lock:
            stats["skipped"] += 1
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ptbxl-research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size == 0:
                raise OSError("empty file")
            os.replace(tmp, dest)
            with lock:
                stats["ok"] += 1
            return True
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                with lock:
                    stats["failed"].append(f"{url} :: {e}")
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rec_dir = data_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    db = load_database(data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv")
    print(f"{len(db)} records with a diagnostic superclass")

    jobs = []
    for _, row in db.iterrows():
        name = Path(row["filename_hr"]).name  # e.g. 00001_hr (globally unique)
        for ext in (".dat", ".hea"):
            url = BASE + row["filename_hr"] + ext
            jobs.append((url, rec_dir / (name + ext)))

    print(f"{len(jobs)} files to fetch with {args.jobs} workers")
    done = {"n": 0}
    total = len(jobs)

    def one(job):
        fetch(*job)
        with lock:
            done["n"] += 1
            if done["n"] % 1000 == 0:
                print(f"  ... {done['n']}/{total} "
                      f"(ok={stats['ok']} skipped={stats['skipped']} "
                      f"failed={len(stats['failed'])})", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(one, jobs))

    (data_dir / "full_download_failures.txt").write_text("\n".join(stats["failed"]))
    print(f"done: ok={stats['ok']} skipped={stats['skipped']} "
          f"failed={len(stats['failed'])}")
    return 0 if not stats["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
