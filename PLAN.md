# Plan: physics-grounded open-set drone classifier

Branch `physics-openset`. It extends the repo's binary CNN to a radar-agnostic, multi-head, open-set model, trained with simulated outlier drones and self-training. A bladeRF bench is the second radar domain.

Status key: ✅ done · 🟡 partly done / needs a decision · ⏳ not started · 🖥 desktop phase

## What the data showed (before any modelling)

- **Split leakage.** The dataset's `split` field mixes segments of *every* session across train/val/test, 0.1 s apart. `data.session_split()` keeps each session whole in one split, at roughly 61/21/17% of segments. Pigeon and raven have one session each and stay in train.
- **Radar.** 77 GHz, PRF 17 kHz, 150-sweep dwell: ±16.6 m/s unambiguous, 0.22 m/s bins, 8.8 ms dwell. That is less than one rotor revolution, so rotor rate cannot be observed on SAAB data.

## Phases

| # | Phase | Machine | Status | Exit criterion | Result |
|---|---|---|---|---|---|
| 1 | Baseline + data hygiene (E1, E2, 1×150 input, 3-class H1) | Laptop | ✅ | Baseline mean ± std known; leakage measured | 98.43 ± 0.31% → 95.57 ± 0.20% on the session split (~2.9 points of leakage) |
| 2 | Physics simulator + validation against real spectra | Laptop | 🟡 | Simulated and real commercial-drone spectra statistically close | Medians match (mD 0.077 vs 0.073, entropy 0.282 vs 0.281); KS 0.15–0.28. A grid search over SNR, ρ and blade length did not close the gap, so the mismatch is structural (body-line broadening / clutter) |
| 3 | Multi-head model (H1–H3, FiLM, OE, physics + consistency, open-set score) | Laptop | 🟡 | E3 > 86% | Held-out drone → drone 96.3% ✅ (1 seed). **Open-set flag fails:** AUROC 0.65–0.78, 5–14% flagged in every variant |
| 4 | Self-training (P1, students S → M → L, unknown pool) | Laptop | 🟡 | Gain over supervised in P1; no E3 regression | Student L 92.6% from 10% labels ≈ fully labeled 92.8% ✅; E3 for students not run yet |
| 5 | bladeRF bench (Modes A/B, live inference, tachometer) | Laptop + bladeRF | 🟡 code only | Drone/hand/empty correct; fan → unknown; rotor rate within ±10% | `bladerf_bench.py` written; signal chain tested offline; **hardware path untested** |
| 6 | Scale-up: XL teacher (transformer), P3 cross-dataset (LSS-FMCWR, RDRD), 5 seeds × all ablations, analysis-by-synthesis decoder | 🖥 Desktop | ⏳ | Every claim backed by an ablation | — |
| 7 | Deployment: distil to S, INT8, export | Laptop → Jetson/FPGA | 🟡 | E9 drop < 1% | PTQ INT8 drop 0.14% ✅; ONNX FP32 exported; QAT and on-device runs not done |
| **8** | **Bench inference interface (live display)** | Laptop + bladeRF | ⏳ later | See below | — |
| **9** | **Passive RF detection view (RFUAV-style)** | Laptop + bladeRF | ⏳ later | See below | — |

## Phase 8: bench inference interface (radar view)

Goal: a live display like the RFUAV demo (spectrogram + boxes + label with confidence), driven by **our** radar model on the bladeRF CW returns.

| Element | Content |
|---|---|
| Main panel | Scrolling **micro-Doppler waterfall**: x = time, y = radial velocity (±16.6 m/s in Mode A), colour = dB. One column per slow-time FFT window, built from the same Mode A/B slow-time stream as `bladerf_bench.py` |
| Boxes | One box per dwell (115 ms in Mode A) around the Doppler extent the model used: body line ± the bins above threshold. Colour by H1 class; **dashed** for "unknown UAV" (S_unk ≥ τ); grey for "unresolved" |
| Label (top-left) | `D3 97.2%` / `Unknown UAV (S_unk +2.4)` / `bird 91%` / `human 88%`, from the track-level decision (log-probs summed over the last k dwells, as in E7) |
| Side panel | H3 physics: body velocity, mD fraction, flash count, tip-speed bin, **rotor rate vs tachometer** (Mode A only); a running S_unk plot with the τ line |
| Modes | `live` (bladeRF) and **`replay captures/*.npz`**, so it can be demoed and tested without hardware |
| Output | Optional recording to MP4/GIF for reports |
| Tech | `pyqtgraph` (fast waterfall) or `matplotlib.animation` (no new dependency); inference uses the INT8 or FP32 folded model from `export.py` |

**Exit criterion:** ≥ 10 display updates per second in Mode A on the laptop; the label matches the E8 per-target result on replayed captures.

## Phase 9: passive RF detection view (RFUAV-style)

The RFUAV demo detects a drone's **radio emissions**: remote-control frequency-hopping bursts and video links in the 2.4 / 5.8 GHz ISM bands. It runs object detection on a frequency-vs-time spectrogram of raw IQ (for example, "FutabaT14SG 99.22%"). That is a different sensor from the radar above. It is a useful second channel, because it catches drones whose rotors are hard to see (stealth / ducted), and it misses autonomous drones that don't transmit.

| Item | Plan |
|---|---|
| Hardware | bladeRF 2.0 micro **RX2** listens passively on 2.4 GHz (up to 56 MHz bandwidth) while RX1/TX1 run the 5.8 GHz CW radar. The radar carrier stays at 5.8 GHz so that it doesn't blind the 2.4 GHz receiver |
| Input | STFT spectrogram of raw IQ (frequency × time image), same layout as RFUAV |
| Model | Small detector (YOLO-style or a CNN classifier on burst crops) trained on the public **RFUAV dataset**; INT8 like phase 7 |
| Output | Boxes on each burst + controller/drone model label + confidence (the screenshot's interface) |
| Fusion | Track-level: radar H1/S_unk + RF detection → "drone, known type" / "drone, silent (no RF)" / "RF only (controller, no rotor seen)" |

**Exit criterion:** RF bursts of the bench drone's own controller are boxed and labelled; the fused output separates a silent drone (RF off) from a transmitting one.

## Files

| File | Contents |
|---|---|
| `config.py` | Radar parameter sets (SAAB, bladeRF A/B), waveform vector, loss weights, sizes |
| `data.py` | SAAB loading, session split, 1×150 spectra, label-free measurements, augmentations |
| `physics_sim.py` | Rotor/body/noise simulator, outlier families; `python physics_sim.py` → `dataset/sim_*.npz` |
| `model.py` | FiLM CNN, heads, open-set score, repo baseline |
| `train.py` | Supervised training, Noisy Student loop; `python train.py` → `models/saab.pt`, `models/bench.pt` |
| `evaluate.py` | SIM, E1–E7, E6 ablations, P1 |
| `bladerf_bench.py` | Capture (Modes A/B), live inference, E8 evaluation |
| `export.py` | FiLM folding, INT8 PTQ, ONNX export (E9) |

## Deviations from the original spec

- **Flash count** is counted for one blade. Summed over all blades it is "2+" in 99.9% of samples (4 rotors × 2 blades always flash many times in 8.8 ms), so that target carries no information.
- **Simulated drones in H1** are down-weighted (`sim_h1 = 0.1`). At full weight, seagull false alarms rose from 2.5% to 30% (`logs/sim_h1_sweep.txt`).
- **SupCon** (optional in the spec) is not implemented.
- **Teacher size:** on the laptop the P1 teacher is S; the XL teacher is phase 6.
- **E4 does not apply to the 5×150 ablation**, because its maps are precomputed from clean data.
