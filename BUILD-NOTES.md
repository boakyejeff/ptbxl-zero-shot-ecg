# Build notes — ptbxl-zero-shot-ecg

## What was downloaded (all from PhysioNet, PTB-XL v1.0.3, CC BY 4.0)

| URL | local | size |
|---|---|---|
| https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv | data/ptbxl_database.csv | 6,594,879 bytes |
| https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv | data/scp_statements.csv | 9,720 bytes |
| https://physionet.org/files/ptb-xl/1.0.3/records500/00000/ (dir listing) | — | filenames `<5-digit>_hr.dat` / `.hea` |
| 200 records (`<id>_hr.dat` + `<id>_hr.hea`), 40 per superclass × 5 | data/records/ (gitignored) | ~24 MB (120,000 bytes per .dat) |

Sample manifest: `data/sample_manifest.csv` (gitignored with data/; regenerate via
`scripts/download_sample.py`).

Class balance in full metadata (dominant-superclass mapping): NORM 9243, MI 4049,
CD 3431, STTC 3360, HYP 1305, 411 records with no diagnostic superclass dropped.
Mapping recipe: `ast.literal_eval` on `scp_codes`, aggregate likelihoods per
superclass via `scp_statements.csv` `diagnostic_class`, take argmax (standard
PTB-XL approach).

## Bugs hit and fixes

1. **PEP 668 externally-managed environment**: `pip install` refused. Fixed with
   `--break-system-packages`.
2. **pip wanted to uninstall debian numpy 1.26.4** (RECORD file missing) when
   installing scikit-learn/scipy. Fixed by installing with `--no-deps` and
   adding the small pure-python deps (`joblib`, `threadpoolctl`, `narwhals`)
   separately. Result: scikit-learn 1.9.1 on system numpy 1.26.4 / scipy 1.11.4.
3. **torch CPU wheel (~190 MB) exhausted /tmp** (512 MB tmpfs) mid-download and
   the proxy kept dropping the connection. Decision: dropped torch entirely —
   sklearn `MLPClassifier` covers the "small MLP" design option, which is what
   the task allowed as option (a).
4. **wfdb install stalled on the proxy**; wrote a native WFDB format-16 reader
   (`src/dataio.py::_read_native`, pure numpy: `<i2` sample-major, per-lead
   ADC gain from the `.hea`). `read_record` uses `wfdb` when importable,
   otherwise the native reader. (wfdb never got installed; native reader used.)
5. **`NearestCentroid` has no `.transform()`** in sklearn 1.9 — baseline OOD
   distances computed via `pairwise_distances(X, clf.centroids_)`.
6. **Feature dim miscount**: `_lead_features` returns 20 (not 19) features/lead
   → 240 total. Docstrings/asserts updated.
7. **PhysioNet is very slow** (~1–2 MB/min through this proxy): serial download
   of 400 files would have taken hours. Used 6-way parallel curl
   (`scripts/download_parallel.sh`) plus an idempotent resume fix-up
   (`scripts/fixup_download.sh`, checks `.dat` == 120000 bytes).
8. **Synthetic-data sanity run**: max-softmax OOD AUROC ≈ 0.0 on toy data where
   each class had a huge lead-specific shift — the 4-way MLP becomes *more*
   confident on the novel class than on ID data. Metric direction kept as-is
   (higher max-softmax = more ID-like); AUROC ≈ 0 is informative, not a bug.

## Honest numbers

Primary run: 200 records (40/superclass), 5 LOSO folds, `n_support=5`, seed 0.

| held-out | OOD-AUROC (max-softmax) | OOD-AUROC (energy) | 5-shot acc (held-out) | 5-shot acc (overall) | baseline OOD-AUROC | baseline 5-shot acc |
|---|---|---|---|---|---|---|
| NORM | 0.6391 | 0.6813 | 0.3429 | 0.2651 | 0.3516 | 0.2000 |
| MI   | 0.6313 | 0.5469 | 0.2286 | 0.2771 | 0.4109 | 0.0000 |
| STTC | 0.3755 | 0.4589 | 0.2571 | 0.3012 | 0.3844 | 0.0286 |
| CD   | 0.4944 | 0.5653 | 0.2286 | 0.3095 | 0.5531 | 0.4857 |
| HYP  | 0.4194 | 0.4061 | 0.4000 | 0.2857 | 0.7541 | 0.0286 |

Means ± std (5 folds): OOD-AUROC(ms) **0.5119±0.1203**, OOD-AUROC(en)
**0.5317±0.1058**, 5-shot held-out **0.2914±0.0767**, 5-shot overall
**0.2877±0.0179**, baseline OOD-AUROC **0.4908±0.1660**, baseline 5-shot
**0.1486±0.2044**. (Chance: 0.50 / 0.20.)

Robustness: same 200 records, different eval seeds → seed 1:
OOD-AUROC(ms) 0.5168±0.0574, 5-shot 0.3543±0.1042, baseline 5-shot 0.1143±0.1294;
seed 2: OOD-AUROC(ms) 0.5265±0.1225, 5-shot 0.3486±0.0935, baseline 5-shot
0.4229±0.3053. Conclusion stable: OOD ≈ chance, few-shot modestly above chance
and consistently above the (high-variance) baseline.

Diagnostics (why the null): 4-way ID test accuracy per fold (seed 0):
NORM-held-out 0.333, MI 0.479, STTC 0.417, CD 0.469, HYP 0.306 (chance 0.25;
train acc up to 0.93 → memorization). Ablations: hidden=32/α=1.0 → 0.363;
hidden=32/α=10 → 0.322; hidden=96/α=1.0 → 0.413; logreg C=0.1 → 0.455;
LDA → 0.400; PCA20+MLP → 0.356. Nothing escapes ~0.3–0.46: the binding
constraint is n≈112 training records for a 240-dim 4-way problem, not the
optimizer.

## Runtime notes

- Feature extraction: ~8 s for 150 signals on CPU (~50 ms/signal); 200 real
  records ≈ 12 s.
- Full 5-fold LOSO (`python -m src.run_experiment`): ~10–15 s on CPU
  (5 small MLP fits dominate).
- `python -m src.smoke_test`: ~3 s end-to-end (24 records).
- Download: PhysioNet ~1–2 MB/min via this proxy; 200 records (25 MB) took
  ~25 min serial-equivalent, ~10 min with 6 parallel curl jobs.
