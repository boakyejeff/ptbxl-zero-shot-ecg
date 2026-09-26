# Cross-Superclass Zero-Shot / Few-Shot Generalization on PTB-XL ECGs

![Project cover](assets/cover.png)
## Novelty claim

Standard PTB-XL benchmarks train and test on the **same** 5 diagnostic
superclasses (NORM, MI, STTC, CD, HYP) with random or stratified splits, so a
model can succeed by memorizing class-conditional waveform statistics it has
already seen. This repo asks a harder, clinically motivated question:

> **If an entire diagnostic superclass was never seen during training, can an
> ECG embedding still (a) flag it as out-of-distribution, and (b) classify it
> from just 5 labeled examples?**

We use a **leave-one-superclass-out (LOSO)** protocol — 5 folds, each holding
out a full superclass for evaluation — and test two capabilities on the held-out
class: **OOD detection** (max-softmax / energy AUROC of known-class test samples
vs held-out-class samples) and **5-shot nearest-centroid classification** in the
learned embedding space. A raw-signal nearest-centroid baseline runs the same
protocol, so every number is a *delta over trivial*, not an absolute.

This differs from the PTB-XL leaderboard (Strodthoff et al., 2020) and from
typical "zero-shot ECG" papers that hold out *statements* within seen
superclasses: here the held-out concept is an entire top-level diagnostic
category, so there is no within-superclass leakage to exploit.

## Method (CPU-feasible by design)

- **Data**: PTB-XL v1.0.3 (PhysioNet, CC BY 4.0). ~200 records sampled from
  `records500/00000/`, balanced across the 5 superclasses; dominant superclass
  per record = argmax of likelihoods aggregated per superclass (standard recipe).
- **Features**: 20 hand-crafted features per lead × 12 leads = 240 dims
  (moments, robust ranges, zero-crossing rate, Hjorth mobility/complexity,
  spectral centroid/dominant frequency). Rationale: with ~200 records a 1D-CNN
  trained 5× (once per LOSO fold) would be data-starved and slow; the MLP-on-
  features variant trains each fold in under a minute on CPU and is a fair test
  of whether *simple* embeddings generalize across superclasses.
- **Model**: `StandardScaler` → MLP (1 hidden layer, 96 units, ReLU, Adam,
  early stopping) trained 4-way on the known superclasses per fold.
  Embeddings = penultimate-layer activations.
- **Baseline**: raw 12-lead signal downsampled to 125 samples/lead (1500 dims),
  standardized, plain `NearestCentroid` — no learning at all.
- **Splits**: known classes split train/test with patient-grouped shuffling
  where possible (GroupShuffleSplit, fallback to stratified; see Limitations).

## Results

Real numbers from a real run on the downloaded sample (200 records, 40/class,
5 LOSO folds, `n_support=5`, seed 0). Chance levels: OOD AUROC 0.50, 5-shot
accuracy 0.20 (5-way).

| held-out | OOD AUROC (max-softmax) | OOD AUROC (energy) | 5-shot acc (held-out) | 5-shot acc (overall) | baseline OOD AUROC | baseline 5-shot acc |
|---|---|---|---|---|---|---|
| NORM | 0.639 | 0.681 | 0.343 | 0.265 | 0.352 | 0.200 |
| MI | 0.631 | 0.547 | 0.229 | 0.277 | 0.411 | 0.000 |
| STTC | 0.376 | 0.459 | 0.257 | 0.301 | 0.384 | 0.029 |
| CD | 0.494 | 0.565 | 0.229 | 0.310 | 0.553 | 0.486 |
| HYP | 0.419 | 0.406 | 0.400 | 0.286 | 0.754 | 0.029 |
| **mean ± std** | **0.512 ± 0.120** | **0.532 ± 0.106** | **0.291 ± 0.077** | **0.288 ± 0.018** | **0.491 ± 0.166** | **0.149 ± 0.204** |

**Takeaway (honest null on OOD, weak positive on few-shot):**
- *OOD detection fails*: max-softmax and energy AUROCs sit at chance
  (0.51–0.53). A 4-way MLP trained on 4 superclasses is not reliably less
  confident on the 5th — it confidently misattributes unseen pathology to known
  classes. This held across two extra evaluation seeds (0.52, 0.53).
- *5-shot helps a little*: nearest-centroid in the learned embedding reaches
  0.29 on held-out queries vs 0.15 for the raw-signal baseline and 0.20 chance.
  The embedding carries *some* transferable structure, but not much.
- *Diagnosis*: the base 4-way classifier itself only reaches 0.31–0.48 test
  accuracy (chance 0.25) with ~112 training records — train accuracy hits 0.93,
  i.e. it memorizes. Stronger regularization (α=1, 10; hidden=32) and
  PCA/linear alternatives (logreg 0.46, LDA 0.40) barely move the needle: at
  this sample size the problem is data scarcity, not the optimizer. The null
  result is therefore informative, not a bug: **cross-superclass generalization
  needs either far more data or a stronger inductive bias than a small MLP on
  hand-crafted features.**

## How to run

```bash
pip install -r requirements.txt        # wfdb, numpy, pandas, scikit-learn, scipy
python scripts/download_sample.py     # ~200 records, ~25 MB, from PhysioNet
python -m src.smoke_test              # end-to-end sanity check, ~1-2 min
python -m src.run_experiment          # full 5-fold LOSO, writes results/
```

`results/loso_results.csv` and `results/summary.json` hold the per-fold and
aggregate numbers. Features are cached to `results/features_cache.npz` so
re-runs skip extraction.

## Limitations (honest)

- **Small sample** (~200 records): each fold trains on ~110–130 records. MLP
  hyper-parameters are conservative (early stopping, L2), but overfitting risk
  remains; treat absolute numbers as indicative, deltas vs. baseline as the
  signal.
- **Patient grouping is best-effort**: `stratified_sample_ids` prefers distinct
  patients, but the 00000 slice is small, so some folds fall back to stratified
  shuffling. Any residual same-patient leakage would *inflate* ID-side metrics,
  not the held-out-class metrics that matter here.
- **Held-out classes are "seen" distributionally**: e.g. an MI held out still
  shares acquisition devices/sites with training data; this is a *semantic*
  hold-out, not a domain shift.
- **Feature choice is deliberately simple**: a contrastive or 1D-CNN embedding
  trained on the full 21k PTB-XL records would likely do better; this repo tests
  the *protocol*, not the SOTA model.

## Repo layout

```
src/meta.py          metadata -> dominant superclass, stratified sampling
src/dataio.py        WFDB waveform reading
src/features.py      hand-crafted per-lead features + raw-signal baseline features
src/model.py         EmbeddingMLP (sklearn, CPU)
src/evaluate.py      LOSO protocol: OOD AUROC, 5-shot kNN, baselines
src/run_experiment.py  CLI driver, writes results/
src/smoke_test.py    fast end-to-end sanity check
scripts/download_sample.py  balanced sample download from PhysioNet
```

## Data provenance

PTB-XL v1.0.3, PhysioNet, CC BY 4.0 (Wagner et al., PhysioNet 2020).
Waveforms are downloaded at build time, never committed (see `.gitignore`).
