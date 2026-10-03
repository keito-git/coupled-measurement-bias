# Spurious Effect Modification from Crowdsourced Annotations in Event Sequence Mining: Coupled Measurement Bias and a Diagnostic Test

Code and result files for the paper. The code

* fits a discrete-time, arrival-conditioned Hawkes-type mark model (DT-AMHP) in which an ambiguity modifier
  (the vote entropy of the crowd labels) modulates excitation through a parameter gamma;
* derives and checks numerically the population bias of gamma when the event label is the majority vote
  and the modifier is the vote entropy (Proposition 1);
* fits i.i.d. and AR(1) latent-difficulty annotator models to the raw crowd votes;
* simulates null distributions of gamma_hat under gamma = 0 (S1 with entropy-linked accuracy, S2 with the AR(1) annotator
  model, and the fully generative pipeline C0) and uses them as a diagnostic test on EmotionLines
  (Friends, EmotionPush) and M3ED;
* evaluates soft-mark, latent-mark, calibration (Neyman belt) and MC-SIMEX corrections;
* writes every number reported in the paper as a LaTeX macro (`analysis/numbers.tex`).

## Layout

```
README.md  LICENSE  requirements.txt
src/                         library
  config.py                  paths (DATA_ROOT, RESULTS_ROOT) and process settings
  prepare_m3ed.py            M3ED annotation.json -> per-utterance vote table (parquet)
  estimator_dt.py            hard-mark DT-AMHP likelihood, gradient and fitter
  estimator_softmark.py      soft-mark estimator
  dtsim_core.py              EmotionLines loading, within-cell residualisation, bounded fit
  dtsim_fits.py              within-cell / raw-H fits and aggregation helpers
  dtsim_kernels.py           numba simulators (accuracy models, votes, DT-AMHP generator)
  el_wide_nulls.py           EL-wide null simulators S1/S2/C0, decision table, power (E35/E38/E39)
  corpus_nulls.py            per-corpus nulls for Friends, EmotionPush, M3ED (E33c/E40)
  latent_mark.py             latent-mark correction and matched data-generating process (E34e/E34f/E37)
experiments/<ID>/            one folder per experiment: scripts and their result files
analysis/
  two_sided_pvalues.py       median-centred two-sided p-values from the checkpoints
  make_numbers.py            all paper numbers -> analysis/numbers.tex
  make_tkde_figures.py       Fig. "pipeline" and Fig. "calibration" -> analysis/figures/
```

All result JSON files (aggregates and per-replication checkpoints, each below 5 MB) are included, so
`analysis/` runs without re-running any simulation.

## Setup

Python 3.11 (tested with 3.11.3).

```
pip install -r requirements.txt
```

Paths are set by two environment variables:

* `DATA_ROOT` (default `<repo>/data`): raw corpora, see below.
* `RESULTS_ROOT` (default `<repo>/experiments`): experiment outputs; scripts read their inputs from the
  folders of earlier experiments under the same root.
* `N_WORKERS` (optional): number of worker processes of the simulation scripts (the results do not depend on it).

## Data

The corpora are not redistributed. Place them as follows:

```
DATA_ROOT/raw/emotionlines/Friends/friends.json
DATA_ROOT/raw/emotionlines/EmotionPush/emotionpush.json
DATA_ROOT/raw/m3ed/annotation.json
```

* EmotionLines, EmotionX 2019 release: `friends.json` and `emotionpush.json` (the non-augmented files of the
  training archives `2019_Train_Friends.zip` and `2019_Train_EmotionPush.zip`, which unpack to `Friends/` and
  `EmotionPush/`). Each utterance has an `annotation` string with the vote counts of five annotators over
  (neutral, joy, sadness, fear, anger, surprise, disgust).
  C.-C. Hsu, S.-Y. Chen, C.-C. Kuo, T.-H. Huang, L.-W. Ku, "EmotionLines: An Emotion Corpus of Multi-Party
  Conversations," LREC 2018; B. Shmueli, L.-W. Ku, "SocialNLP EmotionX 2019 Challenge Overview: Predicting
  Emotions in Spoken Dialogues and Chats," arXiv:1909.07734, 2019.
* M3ED: `annotation.json` of the official release.
  J. Zhao, T. Zhang, J. Hu, Y. Liu, Q. Jin, X. Wang, H. Li, "M3ED: Multi-modal Multi-scene Multi-label
  Emotional Dialogue Database," ACL 2022, pp. 5699-5710.

Then build the M3ED vote table (`DATA_ROOT/processed/m3ed_softlabels.parquet`):

```
python src/prepare_m3ed.py
```

Only utterances with three valid annotator labels are used by the experiments.

## Reproducing the paper

Numbers, tables and figures from the included result files:

```
python analysis/two_sided_pvalues.py
python analysis/make_numbers.py          # -> analysis/numbers.tex
python analysis/make_tkde_figures.py     # -> analysis/figures/fig_pipeline.pdf, fig_calibration.pdf
```

Re-running the experiments (each script resumes from its checkpoints if they exist; delete or move the
checkpoints of an experiment to recompute it). Order of dependencies:

```
python src/prepare_m3ed.py
python experiments/E29/run_E29.py
python experiments/E30/run_E30.py && python experiments/E30/run_E30b_m3ed.py
python experiments/E27e/run_E27e.py
python experiments/E28/run_E28.py
python experiments/E31/m5_analytic_bias.py      # also m5b_homogeneous.py, m5c_unimodality.py, m5d_heterogeneous.py
python experiments/E32/run_E32.py && python experiments/E32/run_E32_followup.py
python experiments/E35/run_E35.py && python experiments/E35/posthoc_power_vs_c0.py
python experiments/E38/run_E38.py
python experiments/E39/run_E39.py
python experiments/E40/run_E40.py && python experiments/E33c/run_E33c.py
python experiments/E36/run_E36.py && python experiments/E36/run_E36b_norefine.py
python experiments/E34e/run_E34e.py
python experiments/E34f/run_E34f.py
python experiments/E37/run_E37.py && python experiments/E37/summarize_e37_extra.py
```

| Folder | What it computes | Paper |
|---|---|---|
| E31 | Proposition 1: pseudo-true gamma vs Monte-Carlo MLE (`m5`), homogeneous annotators (`m5b`), unimodality and a second downstream model (`m5c`), annotator heterogeneity (`m5d`) | Sec. 3, Table "proposition_numerical"; Supplement, Tables "supp_unimodal", "supp_het" |
| E29 | i.i.d. annotator model: EM fit, bootstrap SEs, gate G3, simulator check | Sec. 5 Table "corpora"; Supplement, Table "annotator_fit", G3 recovery test |
| E30 | AR(1) annotator model (EM fit; M3ED by direct maximisation in `run_E30b_m3ed.py`) | Sec. 4-5 (S2 simulator, annotation statistics) |
| E27e | Hard-mark nulls (Scheme A', independent and linked accuracy) and observation-pipeline estimates (Scheme C') | Sec. 5 Table "null_hard"; Supplement, Tables "null_hard_indep", "power_obs" |
| E28 | Soft-mark estimator under the Scheme A' nulls, gate G1 | Sec. 5 (soft labels); Supplement, Table "supp_g1" |
| E35 | Decision table of 4 estimators x S1/S2/C0 (B = 200), power, cross-simulator type-I error, S1' entropy sensitivity; `posthoc_power_vs_c0.py`: power against the C0 band | Sec. 5 Table "estimator_null_full"; Supplement, Tables "supp_cross", "supp_power", S1 entropy sensitivity |
| E38 | Primary nulls (hard_wc under S2 and C0) with B = 2000 | Sec. 5 Table "estimator_null_full" |
| E39 | Posterior predictive check of S1/S2/C0 | Supplement, Table "supp_ppc" |
| E33c, E40 | Per-corpus diagnostic (Friends, EmotionPush, M3ED); S1 with B = 200 (E33c), S2 with B = 2000 (E40) | Sec. 5 Table "corpus_diag" |
| E32 | Calibration map, Neyman belt, gate G2, oracle check; follow-up with an extended grid | Sec. 5 Fig. "calibration"; Supplement, Table "calibration", G2 follow-up |
| E36 | Oracle over-shoot by modifier definition and sample size; `run_E36b_norefine.py`: generator without refinement | Supplement, Table "supp_oracle" |
| E34e, E34f | Latent-mark correction under the matched DGP; decomposition of its over-shoot | Supplement, Table "supp_latmark" |
| E37 | MC-SIMEX | Supplement, Table "supp_simex", MC-SIMEX section |

The p-values in Table "null_hard" and the S1 rows of Table "corpus_diag" are the median-centred two-sided
values of `analysis/two_sided_pvalues.py`; the other experiments compute the same definition directly.

## Expected runtimes

Wall-clock times of the original runs (laptop-class CPU; workers in parentheses):
E27e about 14 h (10), E28 1.3 h (4), E29 1 h (4), E30 17 h (4), E30b 1.7 h (1), E32 11 h (3) plus
follow-up 2.5 h (4), E33c 0.35 h (4), E35 1.3 h (10), E36 3.1 h (3), E38 1.9 h (6), E39 2 min (2),
E40 1.8 h (3), E34e 15 min (3), E34f, E37 and E36b well under 1 h (3). E31: seconds to 3 min per script.
`analysis/` takes seconds.

## Notes on reproducibility

* Every replication has its own seed (written in the scripts), so results do not depend on the number of
  workers. Re-running single replications of E27e, E28, E32, E33c, E34e, E34f, E35, E36, E37, E38 and E40
  with this code reproduces the stored checkpoint values exactly; the E31 scripts reproduce their result
  files exactly. Pooled summaries can differ in the last floating-point digit because replications finish
  in arbitrary order, and the EM fits of E29/E30 can differ at the 1e-10 level across BLAS builds.
* The simulated-copy summaries in `E29_simcheck.json`, `E29_ppc.json` and `E30_ppc.json` were generated with
  a seed offset based on Python's salted string hash; the scripts now use a deterministic offset, so these
  simulated summaries (not used for any reported number) can differ slightly on re-run. Real-data
  statistics and model fits are unaffected.
* `analysis/make_numbers.py` reproduces the macro file of the paper byte for byte from the included results.

## Citation

```
@article{spurious2026,
  title   = {Spurious Effect Modification from Crowdsourced Annotations in Event Sequence Mining:
             Coupled Measurement Bias and a Diagnostic Test},
  author  = {Keito Inoshita and Atsushi Takenaka},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  note    = {under review},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The corpora are subject to their own licences.
