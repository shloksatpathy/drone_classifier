# Results: physics-openset branch (laptop, RTX 3050 4 GB)

Raw output: `logs/results.txt`, `logs/sim_h1_sweep.txt`. E1/E2 use 5 seeds. **E3–E7, ablations and P1 use one seed**, so differences of a few points between variants are within run-to-run noise until the desktop phase repeats them with 5 seeds.

## E1 / E2: baseline and leakage

| | Accuracy | Balanced accuracy |
|---|---|---|
| E1 repo model, dataset split | 98.43 ± 0.31% | 98.35 ± 0.09% |
| E2 repo model, **session split** | 95.57 ± 0.20% | 96.01 ± 0.25% |

The dataset's per-segment split inflates accuracy by **~2.9 points**. All later numbers use the session split.

## Phase 2: simulator vs. real spectra (KS statistic, lower = closer)

| Statistic | Real drone median | Commercial sim (median, KS) | Exotic sim (median, KS) | Bird vs. drone KS (scale reference) |
|---|---|---|---|---|
| mD fraction | 0.073 | 0.077, 0.175 | 0.095, 0.111 | 0.114 |
| Occupied bins (−20 dB) | 0.047 | 0.033, 0.239 | 0.047, 0.131 | 0.134 |
| Spectral entropy | 0.281 | 0.282, 0.154 | 0.322, 0.140 | 0.234 |
| Peak-to-median (dB) | 35.1 | 32.5, 0.277 | 31.3, 0.217 | 0.236 |

Medians match; the distributions are about as far apart as real birds are from real drones. Stealth samples are deliberately far off (KS 0.36–0.68). The exit criterion is **not met**: the simulator is good enough to help (E3, E4) but not to be called realistic.

## Main model and ablations (session split, test set)

| Variant | 3-class acc | 3-class bal. acc | Drone/not acc | Seagull → drone | **E3** held-out → drone | E3 unknown AUROC | E3 flagged unknown |
|---|---|---|---|---|---|---|---|
| **full** (sim_h1 = 0.1) | 92.8% | 0.785 | 95.2% | 17.9% | **96.3%** | 0.710 | 5.7% |
| −FiLM | 92.9% | 0.807 | 95.4% | 18.3% | 90.8% | 0.707 | 5.4% |
| −OE | 93.9% | 0.825 | 96.1% | 11.7% | 92.9% | 0.737 | 5.6% |
| −H3 (no physics, no consistency) | 92.0% | 0.726 | 93.8% | 33.6% | 94.9% | 0.670 | 5.5% |
| −L_cons | 93.1% | 0.798 | 95.4% | 17.6% | 96.6% | 0.653 | 5.6% |
| −sim | 93.4% | **0.887** | 95.1% | **2.3%** | 91.0% | 0.748 | 8.0% |
| 5×150 input (−sim) | **94.7%** | **0.911** | **96.2%** | 14.2% | 90.2% | **0.782** | **13.5%** |

Reference: Karlsson et al. report 78 → 86% for out-of-distribution drones with synthetic data (their protocol, not identical to this one).

### E4: SNR sweep (3-class accuracy / drone-vs-not AUROC)

| Added noise | full | −sim |
|---|---|---|
| none | 92.8% / 0.984 | 93.4% / 0.992 |
| 30 dB | 91.8% / 0.980 | 92.2% / 0.988 |
| 20 dB | 89.9% / 0.971 | 88.2% / 0.982 |
| 10 dB | 82.8% / 0.907 | 70.6% / 0.930 |
| 0 dB | **76.7%** / 0.705 | 39.3% / 0.759 |

### E7: track level (3-class accuracy vs. number of 10 Hz scans)

| Scans | 1 | 3 | 5 | 10 |
|---|---|---|---|---|
| full | 92.8% | 95.8% | 96.6% | **97.5%** |
| −sim | 93.4% | 95.4% | 95.9% | 96.4% |

## P1: self-training (10% labels: 4,441 labeled, 40,025 unlabeled)

| Model | 3-class acc | Bal. acc | Seagull → drone |
|---|---|---|---|
| Teacher S, 10% labels | 90.9% | 0.776 | 19.0% |
| Student S | 88.3% | 0.770 | 1.4% |
| Student M | 91.0% | 0.709 | 15.8% |
| Student L | **92.6%** | **0.795** | 7.6% |
| Reference: full model, 100% labels | 92.8% | 0.785 | 17.9% |

Pseudo-labels kept: ~21–22k per round at 99.0% accuracy. Unknown pool: 525 → 74 → 365 segments; the clusters are mostly gulls/heron and D1, not a clean "unknown drone" group.

## E9: INT8

| | 3-class acc |
|---|---|
| Original | 92.28% |
| FiLM folded, FP32 | 92.28% |
| INT8 (FX PTQ, x86) | 92.14% (**−0.14%**, H1 agreement 99.5%) |

## What this says

1. **Leakage is real:** ~3 points on the repo baseline.
2. **Simulated outliers buy recall on unseen drones and at low SNR**, and pay for it with bird false alarms. E3 rises from 91.0% to 96.3%, and accuracy at 0 dB from 39% to 77%. Seagull false alarms go from 2.3% to 17.9%. `sim_h1` is the knob (sweep: 0 → 92.4% / 6.6%, 0.1 → 96.8% / 13.7%, 0.3 → 94.7% / 28.6%).
3. **The open-set flag does not work yet.** AUROC is 0.65–0.78 and only 5–14% of held-out drones get flagged in every variant. Held-out drone types land inside the known-drone clusters, so "drone" works but "*which* drone / unknown" does not. This is the main open problem. Next steps: SupCon on z, per-type covariance, and scoring on the track (E7-style) rather than per segment.
4. **Physics heads help:** without H3, seagull false alarms reach 33.6% and track-level gains shrink. OE did not help (−OE is better on every metric except E3). FiLM cannot matter on SAAB-only data (w is constant), so the −FiLM gap is seed noise. FiLM is only tested once bench data exists.
5. **The 5×150 input beats 1×150** on balanced accuracy and open-set AUROC. Range context matters on SAAB; the 1×150 input is the price of radar-agnosticism.
6. **Self-training** with student L recovers the fully labeled result from 10% of the labels (92.6% vs 92.8%).
7. **Tracks:** accumulating 10 scans reaches 97.5%.
8. **INT8** is essentially free.
