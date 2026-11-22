# Drone classifier (77 GHz FMCW radar)

Classifies radar detections as **drone** or **not drone** (birds, humans) using a small CNN on Doppler spectra.

## Dataset

[SAAB SIRS 77 GHz FMCW radar dataset](https://zenodo.org/records/5845259): 130 recording sessions of six drones (D1–D6), walking/running humans and several bird species. Download `data_SAAB_SIRS_77GHz_FMCW.npy` into `dataset/`.

## Run

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python drone_classifier.py dataset/data_SAAB_SIRS_77GHz_FMCW.npy
```

The best epoch (by validation balanced accuracy) is saved to `drone_cnn.pt`.

## Method

1. **Features**: each segment (5 range cells × 256 sweeps) is cropped to the 150 sweeps around the beam centre, then an FFT across sweeps gives a Doppler (speed) spectrum in dB. Rotor blades spread energy across many speeds; birds and people stay narrow.
2. **Normalization**: each sample is standardized on its own, so the model learns spectrum shape rather than range or signal strength.
3. **Model**: 3 conv blocks (16/32/64 filters, BatchNorm, ReLU, max-pool on the Doppler axis) → global average pool → linear layer with 2 outputs. About 39k parameters.
4. **Training**: Adam (lr 1e-3), class-weighted cross-entropy, 15 epochs, using the dataset's own train/val/test split.

## Results (test set, 8,000 segments)

| | |
|---|---|
| Accuracy | 98.7% |
| Drone recall | 98.9% |
| Non-drones flagged as drone | 1.9% |
| Hardest target | seagull (7.2% flagged as drone) |

Full output: [`train_log.txt`](train_log.txt).

## Limitations

- No cars or ground vehicles in the dataset.
- Very few pigeon and raven samples.
- One radar at one site.
