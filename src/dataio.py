"""WFDB waveform reading for the PTB-XL 500 Hz records."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import wfdb
except ImportError:  # pragma: no cover
    wfdb = None


def _read_native(record_path: str | Path) -> np.ndarray:
    """Minimal WFDB format-16 reader (pure numpy, no wfdb dependency).

    PTB-XL 500 Hz records are 16-bit little-endian, sample-major multiplexed
    (all 12 leads per sample instant), with per-lead ADC gain in the .hea.
    Validated against wfdb.rdrecord on the downloaded sample: identical output.
    """
    hea_lines = (Path(str(record_path) + ".hea")).read_text().splitlines()
    n_sig, n_samp = (int(v) for v in hea_lines[0].split()[1:4:2])
    gains = [float(line.split()[2].split("(")[0]) for line in hea_lines[1 : 1 + n_sig]]
    raw = np.fromfile(str(record_path) + ".dat", dtype="<i2")
    sig = raw.reshape(n_samp, n_sig).astype(np.float32)
    return sig / np.asarray(gains, dtype=np.float32)


def read_record(record_path: str | Path) -> np.ndarray:
    """Read one PTB-XL record -> float32 array of shape (5000, 12), units mV.

    record_path is the path WITHOUT extension (WFDB convention), e.g.
    data/records/00001_hr
    """
    if wfdb is not None:
        rec = wfdb.rdrecord(str(record_path))
        sig = np.asarray(rec.p_signal, dtype=np.float32)
    else:  # fallback: native format-16 reader
        sig = _read_native(record_path)
    if sig.shape != (5000, 12):
        raise ValueError(f"unexpected shape {sig.shape} for {record_path}")
    return sig


def local_record_path(data_dir: str | Path, filename_hr: str) -> Path:
    """Map a PTB-XL filename_hr (records500/00000/00001_hr) to a local path."""
    name = Path(filename_hr).name  # e.g. 00001_hr
    return Path(data_dir) / "records" / name


def load_signals(
    df,
    data_dir: str | Path,
) -> dict[int, np.ndarray]:
    """Read all waveforms for the rows of df; returns {ecg_id: signal}."""
    out = {}
    missing = []
    for _, row in df.iterrows():
        p = local_record_path(data_dir, row["filename_hr"])
        try:
            out[int(row["ecg_id"])] = read_record(p)
        except FileNotFoundError:
            missing.append(int(row["ecg_id"]))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} records missing under {data_dir}/records, "
            f"e.g. ecg_id={missing[:5]}. Run scripts/download_sample.py first."
        )
    return out
