# Model Report: Drone Classifier (77 GHz FMCW Radar)

## 1. Objective

Binary classification of radar detections as **drone** or **not drone** (birds, walking/running humans). The input is the Doppler spectrum of a short radar segment.

## 2. Dataset

**Source:** [SAAB SIRS 77 GHz FMCW radar dataset](https://zenodo.org/records/5845259) (`data_SAAB_SIRS_77GHz_FMCW.npy`). It holds 130 recording sessions, and each row stores the label, raw data, range, time, split and edge.

- **Positive class (1):** drones D1–D6
- **Negative class (0):** gulls, seagulls, heron, pigeon, raven, human walk and human run
- **Excluded:** `CR` (corner reflector). It is a calibration target, not a real class.

### Samples per label

| Label | Segments |
|---|---:|
| D1 | 6,921 |
| D2 | 9,555 |
| D3 | 10,212 |
| D4 | 8,735 |
| D5 | 12,093 |
| D6 | 11,252 |
| black-headed gull | 4,364 |
| seagull | 2,550 |
| human_walk | 4,529 |
| human_run | 1,499 |
| seagull and black-headed gull | 440 |
| heron | 387 |
| pigeon | 32 |
| raven | 19 |

### Split (from the dataset's own split field)

| Split | Segments | Drone share |
|---|---:|---:|
| Train | 56,588 | 82.6% |
| Validation | 8,000 | 75.0% |
| Test | 8,000 | 75.0% |

## 3. Preprocessing

1. Each segment is reshaped to **5 range cells × 256 FM sweeps**.
2. The 150 sweeps around the beam centre (indices 54–203) are kept, as in the original paper.
3. An FFT along slow time (across sweeps), followed by `fftshift`, gives a **Doppler spectrum**.
4. The magnitude is converted to dB with `20·log10(|X| + 1e-12)`.
5. **Per-sample standardization** (zero mean, unit std over each 5×150 map) removes absolute power. This way the model learns the spectrum's shape and not the range or radar cross-section.
6. Final input tensor: **(N, 1, 5, 150)**, i.e. channel × range × Doppler.

**Why it works:** spinning rotor blades spread energy across many Doppler bins (micro-Doppler). Birds and people produce narrower spectra.

## 4. Model

This is a small 2D CNN with 3 convolutional blocks, global average pooling and a linear classifier. It has **39,106 trainable parameters**. Layer details are in [`layers.md`](layers.md).

```
Input (1×5×150)
 → [Conv 3×5, 16] → BN → ReLU → MaxPool(1×2)   → 16×5×75
 → [Conv 3×5, 32] → BN → ReLU → MaxPool(1×2)   → 32×5×37
 → [Conv 3×5, 64] → BN → ReLU                  → 64×5×37
 → Global Avg Pool → Flatten                   → 64
 → Linear(64 → 2)                              → logits
```

## 5. Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 1e-3 |
| Loss | Cross-entropy, class-weighted (`N / (2 · count_c)`) to offset the 82.6% drone share in training |
| Epochs | 15 |
| Batch size | 256 (shuffled each epoch) |
| Model selection | Checkpoint with the best **validation balanced accuracy** is saved to `drone_cnn.pt` |
| Device | CUDA if available, else CPU |

### Training curve

| Epoch | Last-batch loss | Val balanced acc |
|---:|---:|---:|
| 1 | 0.4026 | 0.8648 |
| 2 | 0.1421 | 0.9409 |
| 3 | 0.0540 | 0.9711 |
| 4 | 0.0288 | 0.9765 |
| 5 | 0.1361 | 0.9313 |
| 6 | 0.0023 | 0.9823 |
| 7 | 0.0059 | 0.9779 |
| 8 | 0.0060 | 0.9480 |
| 9 | 0.0006 | 0.9802 |
| 10 | 0.0077 | 0.9838 |
| 11 | 0.0013 | 0.9812 |
| 12 | 0.1067 | 0.9206 |
| 13 | 0.0534 | 0.9761 |
| **14** | **0.0556** | **0.9857** ← saved |
| 15 | 0.1564 | 0.9761 |

Validation accuracy swings between epochs (for example, it drops to 0.92 at epoch 12). Saving the best checkpoint guards against ending on a bad epoch.

## 6. Results (Test Set, 8,000 Segments)

### Summary

| Metric | Value |
|---|---:|
| Accuracy | **98.68%** |
| Balanced accuracy (macro recall) | 98.48% |
| Macro F1 | 0.9824 |
| Drone recall | 98.87% |
| Drone precision | 99.36% |
| Non-drones flagged as drone (FPR) | 1.90% |

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| not drone | 0.9665 | 0.9810 | 0.9737 | 2,000 |
| drone | 0.9936 | 0.9887 | 0.9911 | 6,000 |

### Confusion matrix

| | Pred: not drone | Pred: drone |
|---|---:|---:|
| **True: not drone** | 1,962 | 38 |
| **True: drone** | 68 | 5,932 |

### Per-target breakdown (share predicted as drone)

| Target | n | Predicted drone |
|---|---:|---:|
| D1 | 1,000 | 97.6% |
| D2 | 1,000 | 100.0% |
| D3 | 1,000 | 97.2% |
| D4 | 1,000 | 99.4% |
| D5 | 1,000 | 99.2% |
| D6 | 1,000 | 99.8% |
| seagull | 305 | **7.2%** |
| seagull and black-headed gull | 45 | 2.2% |
| human_run | 277 | 1.8% |
| black-headed gull | 600 | 1.2% |
| human_walk | 723 | 0.4% |
| heron | 45 | 0.0% |
| pigeon | 4 | 0.0% |
| raven | 1 | 0.0% |

## 7. Observations

- **Drones** are detected reliably. D3 (97.2%) and D1 (97.6%) are the weakest, and D2 reaches 100%.
- **Seagulls** cause the most false alarms (7.2%). Flapping wings likely produce micro-Doppler spread that resembles rotor signatures.
- **Humans** are rarely confused with drones. Running (1.8%) is confused a little more often than walking (0.4%), probably because of the faster limb motion.
- The model is compact (about 39k parameters). It is cheap to run and suitable for real-time or embedded use.

## 8. Limitations

- There are no cars or ground vehicles in the dataset, so performance on them is unknown.
- Pigeon (4) and raven (1) test samples are too few for their results to mean anything.
- All data comes from one radar at one site, so generalization to other hardware or environments is untested.
- Results come from a single training run with no fixed seed. Run-to-run variance is not measured.

## 9. Reproduce

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python drone_classifier.py dataset/data_SAAB_SIRS_77GHz_FMCW.npy
```
