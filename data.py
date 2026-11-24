import numpy as np
import torch
from config import DATA_PATH, N, CROP, RADARS, TYPES


def cls_of(name):
    if name.startswith("D"):
        return 0
    if name.startswith("human"):
        return 2
    return 1


def load_saab(path=DATA_PATH):
    """Per segment: raw complex slow time of the peak range cell (256), the repo's 5x150 map,
    label name, 3-class label, drone type, original split, session id, time."""
    raw = np.load(path, allow_pickle=True)
    out = {k: [] for k in ["raw", "x2d", "name", "split", "session", "time"]}
    for s, row in enumerate(raw):
        name = str(np.ravel(row[0])[0])
        if name == "CR":
            continue
        seg = np.asarray(row[1]).T.reshape(-1, 5, 256)
        peak = (np.abs(seg[:, :, CROP:CROP + N]) ** 2).sum(2).argmax(1)
        out["raw"].append(seg[np.arange(len(seg)), peak].astype(np.complex64))
        spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg[:, :, CROP:CROP + N], axis=2), axes=2)) + 1e-12)
        out["x2d"].append(standardize(spec, (1, 2)).astype(np.float32))
        n = len(seg)
        out["name"].append(np.full(n, name))
        out["split"].append(np.ravel(row[4]).astype(np.int64))
        out["session"].append(np.full(n, s))
        out["time"].append(np.ravel(row[3]))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["cls"] = np.array([cls_of(n) for n in d["name"]])
    d["dtype"] = np.array([TYPES.index(n) if n in TYPES else -1 for n in d["name"]])
    return d


def standardize(x, axis):
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)


def session_split(name, session, seed=0, frac=0.15):
    """Whole sessions go to one split. Per label, shuffled sessions go to test, then val, whenever that
    brings the split closer to frac of the label's segments; the rest train. Labels with < 3 sessions
    stay in train."""
    rng = np.random.default_rng(seed)
    split = np.ones(len(name), dtype=np.int64)
    for lab in np.unique(name):
        sess = np.unique(session[name == lab])
        if len(sess) < 3:
            continue
        total = (name == lab).sum()
        got = {2: 0, 3: 0}
        for s in rng.permutation(sess):
            m = session == s
            for target in (3, 2):
                if abs(got[target] + m.sum() - frac * total) < abs(got[target] - frac * total):
                    split[m] = target
                    got[target] += m.sum()
                    break
    return split


def md_window(radar):
    return max(1, int(round(1.0 / RADARS[radar]["vres"])))


def md_fraction(P, centre, radar):
    """Fraction of spectral power outside +-1 m/s (at least +-1 bin) of the centre bin, circular."""
    h = md_window(radar)
    idx = (centre[:, None] + torch.arange(-h, h + 1, device=P.device)) % N
    inside = P.gather(1, idx).sum(1)
    return 1 - inside / P.sum(1)


def spectra(sig, radar, v_true=None):
    """Complex slow time (B, 150) -> standardized dB spectrum, measured body velocity (normalized to
    [-1, 1) of the velocity axis) and micro-Doppler fraction around the measured peak, or around the
    true body velocity when v_true is given (simulated data)."""
    P = torch.fft.fftshift(torch.fft.fft(sig, dim=1), dim=1).abs() ** 2
    db = 10 * torch.log10(P + 1e-12)
    x = (db - db.mean(1, keepdim=True)) / db.std(1, keepdim=True)
    peak = P.argmax(1)
    v_meas = (peak - N // 2).float() / (N // 2)
    centre = peak if v_true is None else (torch.round(v_true * (N // 2)).long() + N // 2) % N
    return x, v_meas, md_fraction(P, centre, radar)


def augment(raw, gen, noisy):
    """raw (B, L) complex. Clean: centre crop. Noisy: random crop, SNR jitter, circular Doppler roll.
    Returns the 150-sample signal and the roll in bins (to shift simulated velocity targets)."""
    B, L = raw.shape
    if not noisy:
        start = (L - N) // 2 if L > N else 0
        return raw[:, start:start + N], torch.zeros(B, device=raw.device, dtype=torch.long)
    start = torch.randint(0, L - N + 1, (B,), device=raw.device, generator=gen)
    sig = raw.gather(1, start[:, None] + torch.arange(N, device=raw.device))
    snr = torch.rand(B, 1, device=raw.device, generator=gen) * 30
    p = (sig.abs() ** 2).mean(1, keepdim=True) / 10 ** (snr / 10)
    noise = torch.complex(torch.randn(B, N, device=raw.device, generator=gen), torch.randn(B, N, device=raw.device, generator=gen))
    sig = sig + noise * (p / 2).sqrt()
    roll = torch.randint(-20, 21, (B,), device=raw.device, generator=gen)
    sig = sig * torch.exp(2j * torch.pi * roll[:, None] * torch.arange(N, device=raw.device) / N)
    return sig, roll


def mask_bands(x, gen):
    """Zero 1-3 bands of 5-15 bins per sample."""
    B = x.shape[0]
    ar = torch.arange(N, device=x.device)
    for i in range(3):
        on = (torch.rand(B, 1, device=x.device, generator=gen) < 0.5) | (i == 0)
        start = torch.randint(0, N, (B, 1), device=x.device, generator=gen)
        width = torch.randint(5, 16, (B, 1), device=x.device, generator=gen)
        x = x.masked_fill(on & ((ar - start) % N < width), 0)
    return x
