import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

DATA_PATH = sys.argv[1]          # path to data_SAAB_SIRS_77GHz_FMCW.npy
AZ = slice(54, 204)              # 150 FM sweeps around the beam centre (same as the paper)
EPOCHS = 15
BATCH = 256

# ---------- 1. Load and turn each segment into a range-Doppler map ----------
raw = np.load(DATA_PATH, allow_pickle=True)   # 130 x 6: label, data, range, time, split, edge

X, y, split, names = [], [], [], []
for row in raw:
    label = str(np.ravel(row[0])[0])
    if label == "CR":                          # corner reflector = calibration target, not a real class
        continue
    seg = np.asarray(row[1]).T.reshape(-1, 5, 256)[:, :, AZ]      # (n, 5 range cells, 150 sweeps)
    spec = np.fft.fftshift(np.fft.fft(seg, axis=2), axes=2)       # slow-time FFT -> Doppler
    spec = 20 * np.log10(np.abs(spec) + 1e-12)
    n = len(spec)
    X.append(spec.astype(np.float32))
    y.append(np.full(n, int(label.startswith("D"))))              # 1 = drone (D1..D6), 0 = everything else
    split.append(np.broadcast_to(np.ravel(row[4]), n))
    names.append(np.full(n, label))

X, y, split, names = map(np.concatenate, (X, y, split, names))
X = (X - X.mean(axis=(1, 2), keepdims=True)) / X.std(axis=(1, 2), keepdims=True)   # remove absolute power (range/RCS)
X = X[:, None]                                                                        # (N, 1, 5, 150)

print("Samples per label:")
for lab, cnt in zip(*np.unique(names, return_counts=True)):
    print(f"  {lab:35s} {cnt}")
for s, name in [(1, "train"), (2, "val"), (3, "test")]:
    m = split == s
    print(f"{name}: {m.sum()} segments, {y[m].mean():.1%} drones")
