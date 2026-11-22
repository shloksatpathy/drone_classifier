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

# ---------- 2. Model ----------
model = nn.Sequential(
    nn.Conv2d(1, 16, (3, 5), padding=(1, 2)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 2)),
    nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
    nn.Conv2d(32, 64, (3, 5), padding=(1, 2)), nn.BatchNorm2d(64), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

tr, va, te = split == 1, split == 2, split == 3
weights = torch.tensor(len(y[tr]) / (2 * np.bincount(y[tr])), dtype=torch.float32, device=device)
loss_fn = nn.CrossEntropyLoss(weight=weights)          # class imbalance
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])


def predict(Xs):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 2048):
            out.append(model(torch.tensor(Xs[i:i + 2048], device=device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ---------- 3. Train, keep the epoch with best validation balanced accuracy ----------
best = 0
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(model(Xt[idx].to(device)), yt[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    val_bacc = balanced_accuracy_score(y[va], predict(X[va]))
    print(f"epoch {epoch + 1:2d}  loss {loss.item():.4f}  val balanced acc {val_bacc:.4f}")
    if val_bacc > best:
        best = val_bacc
        torch.save(model.state_dict(), "drone_cnn.pt")
