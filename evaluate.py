"""Experiments. Usage: python evaluate.py SIM E1 E2 E3 E4 E5 E6 E7 P1 [seeds=N]"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from config import TYPES, CLASSES, CROP, N
from data import load_saab, session_split, spectra
from model import Net, baseline, score
from train import dev, real_pool, sim_pool, fit, predict, openset_for, noisy_student

args = [a for a in sys.argv[1:] if "=" not in a]
SEEDS = int(dict(a.split("=") for a in sys.argv[1:] if "=" in a).get("seeds", 1))
FLAGS = {"oe", "phys", "cons"}
VARIANTS = {   # name: (model kwargs, use sims, flags)
    "full": (dict(), True, FLAGS),
    "-FiLM": (dict(film=False), True, FLAGS),
    "-OE": (dict(), True, FLAGS - {"oe"}),
    "-H3": (dict(), True, FLAGS - {"phys", "cons"}),
    "-L_cons": (dict(), True, FLAGS - {"cons"}),
    "-sim": (dict(), False, {"cons"}),
    "5x150 (-sim)": (dict(dim=2), False, set()),
}

d = load_saab()
sess = session_split(d["name"], d["session"])
SIMS = [sim_pool(np.load("dataset/sim_saab.npz"), "saab")]


def ms(xs):
    return f"{np.mean(xs):.4f} ± {np.std(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"


# ---------- baseline (repo model, binary, 5x150) ----------
def base_run(split, seed):
    torch.manual_seed(seed)
    y = torch.tensor((d["cls"] == 0).astype(np.int64))
    X = torch.tensor(d["x2d"][:, None])
    tr, va, te = (split == s for s in (1, 2, 3))
    m = baseline().to(dev)
    w = torch.tensor(tr.sum() / (2 * np.bincount(y[tr])), dtype=torch.float32, device=dev)
    lf, opt = nn.CrossEntropyLoss(weight=w), torch.optim.Adam(m.parameters(), lr=1e-3)
    pred = lambda mask: torch.cat([m(X[mask][i:i + 4096].to(dev)).argmax(1).cpu() for i in range(0, mask.sum(), 4096)]).numpy()
    Xt, yt, best, state = X[tr], y[tr], -1, None
    for _ in range(15):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            loss = lf(m(Xt[idx].to(dev)), yt[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            b = balanced_accuracy_score(y[va], pred(va))
        if b > best:
            best, state = b, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        p = pred(te)
    return (p == y[te].numpy()).mean(), balanced_accuracy_score(y[te], p)


def E1():
    r = [base_run(d["split"], s) for s in range(max(SEEDS, 5))]
    print(f"E1 repo baseline, original split: acc {ms([a for a, _ in r])}, balanced acc {ms([b for _, b in r])}")
    return r


def E2():
    r = [base_run(sess, s) for s in range(max(SEEDS, 5))]
    print(f"E2 repo baseline, session split:  acc {ms([a for a, _ in r])}, balanced acc {ms([b for _, b in r])}")


# ---------- our model ----------
def train_variant(name, train_idx, val_idx, seed, known=range(6)):
    kw, use_sims, flags = VARIANTS[name]
    T, V = real_pool(d, train_idx), real_pool(d, val_idx)
    m = fit(Net("S", **kw), T, SIMS if use_sims else [], V, flags, seed)
    return m, openset_for(m, T, V, list(known))


def metrics(m, idx):
    p = predict(m, real_pool(d, idx))
    pred, y = p["h1"].argmax(1), d["cls"][idx]
    gull = d["name"][idx] == "seagull"
    return dict(acc3=(pred == y).mean(), bacc3=balanced_accuracy_score(y, pred),
                bin_acc=((pred == 0) == (y == 0)).mean(), seagull_fa=(pred[gull] == 0).mean()), p


def lodo(name, seed):
    """E3: leave one drone type out of train/val; it is then scored against the test drones of known
    types. Returns per type: % held-out called drone, AUROC of S_unk, % flagged unknown."""
    out = []
    for t in range(6):
        keep = d["dtype"] != t
        m, os = train_variant(name, np.where((sess == 1) & keep)[0], np.where((sess == 2) & keep)[0], seed, [k for k in range(6) if k != t])
        held, known = np.where(~keep)[0], np.where((sess == 3) & keep & (d["dtype"] >= 0))[0]
        ph, pk = predict(m, real_pool(d, held)), predict(m, real_pool(d, known))
        sh, sk = score(os, ph["z"], ph["h2"]), score(os, pk["z"], pk["h2"])
        out.append(((ph["h1"].argmax(1) == 0).mean(),
                    roc_auc_score(np.r_[np.ones(len(sh)), np.zeros(len(sk))], np.r_[sh, sk]),
                    (sh >= os["tau"]).mean()))
        print(f"  {name} held-out {TYPES[t]}: drone {out[-1][0]:.3f}  AUROC {out[-1][1]:.3f}  unknown {out[-1][2]:.3f}", flush=True)
    return np.array(out)


def E3(name="full"):
    r = np.stack([lodo(name, s) for s in range(SEEDS)])          # (seeds, 6, 3)
    mean = r.mean(1)
    print(f"E3 [{name}] held-out drone -> drone {ms(mean[:, 0])}, AUROC {ms(mean[:, 1])}, flagged unknown {ms(mean[:, 2])}")
    return mean


def snr_sweep(m, idx, levels=(None, 30, 20, 10, 0)):
    """E4: add complex noise at the given SNR (vs. signal power of the dwell) to the test segments."""
    g = torch.Generator(device=dev).manual_seed(0)
    res = {}
    if m.dim == 2:            # the 5x150 maps are precomputed from clean data; noise is not applied to them
        return res
    for snr in levels:
        pool = real_pool(d, idx)
        if snr is not None:
            raw = pool["raw"]
            p = (raw[:, CROP:CROP + N].abs() ** 2).mean(1, keepdim=True) / 10 ** (snr / 10)
            pool["raw"] = raw + torch.complex(torch.randn(raw.shape, device=dev, generator=g), torch.randn(raw.shape, device=dev, generator=g)) * (p / 2).sqrt()
        pr = predict(m, pool)["h1"]
        y = d["cls"][idx]
        res[snr] = ((pr.argmax(1) == y).mean(), roc_auc_score(y == 0, pr[:, 0]))
    return res


def main_runs(name):
    """Full-data run(s) on the session split: test metrics, SNR sweep, track level."""
    rows = []
    for s in range(SEEDS):
        m, os = train_variant(name, np.where(sess == 1)[0], np.where(sess == 2)[0], s)
        te = np.where(sess == 3)[0]
        met, p = metrics(m, te)
        rows.append((met, snr_sweep(m, te), track(p["h1"], te)))
    return rows


def track(prob, idx, ks=(1, 3, 5, 10)):
    """E7: sum log-probabilities over k consecutive 10 Hz scans of a session, then decide."""
    res = {}
    for k in ks:
        hit = []
        for s in np.unique(d["session"][idx]):
            j = np.where(d["session"][idx] == s)[0]
            j = j[np.argsort(d["time"][idx][j])]
            lp = np.log(prob[j] + 1e-9)
            c = np.cumsum(np.vstack([np.zeros(3), lp]), 0)
            win = c[k:] - c[:-k]
            hit.extend(win.argmax(1) == d["cls"][idx][j][0])
        res[k] = np.mean(hit)
    return res


def report_main(name, rows):
    g = lambda f: ms([f(r) for r in rows])
    print(f"[{name}] session-split test: acc3 {g(lambda r: r[0]['acc3'])}, bacc3 {g(lambda r: r[0]['bacc3'])}, "
          f"drone/not acc {g(lambda r: r[0]['bin_acc'])}, seagull->drone {g(lambda r: r[0]['seagull_fa'])}")
    for snr in rows[0][1]:
        print(f"  E4 SNR {str(snr):>4s} dB: acc3 {g(lambda r: r[1][snr][0])}, drone AUROC {g(lambda r: r[1][snr][1])}")
    for k in rows[0][2]:
        print(f"  E7 {k:2d} scans: acc3 {g(lambda r: r[2][k])}")


def E4():
    report_main("full", main_runs("full"))


E5 = E7 = E4


def E6():
    for name in VARIANTS:
        t = time.time()
        report_main(name, main_runs(name))
        E3(name)
        print(f"  ({time.time() - t:.0f} s)", flush=True)


def P1():
    for s in range(SEEDS):
        models, _, pool = noisy_student(d, sess, SIMS, FLAGS, seed=s)
        te = np.where(sess == 3)[0]
        for name, m in zip(["teacher S (10% labels)", "student S", "student M", "student L"], models):
            met, _ = metrics(m, te)
            print(f"P1 {name}: acc3 {met['acc3']:.4f}, bacc3 {met['bacc3']:.4f}, seagull->drone {met['seagull_fa']:.4f}")


def SIM():
    """Phase 2: simulated commercial / stealth / exotic spectra vs. real SAAB drones (KS statistic),
    with real birds vs. real drones as the scale of a class-level difference."""
    s = np.load("dataset/sim_saab.npz")

    def stats(sig):
        sig = torch.tensor(sig, device=dev)
        _, _, md = spectra(sig, "saab")
        P = torch.fft.fftshift(torch.fft.fft(sig, dim=1), dim=1).abs() ** 2
        db = 10 * torch.log10(P + 1e-12)
        p = P / P.sum(1, keepdim=True)
        return dict(mD=md, occupied_20dB=(db > db.max(1, keepdim=True).values - 20).float().mean(1),
                    entropy=-(p * torch.log(p + 1e-20)).sum(1) / np.log(N), peak_to_median=db.max(1).values - db.median(1).values)
    real = [{k: v.cpu().numpy() for k, v in stats(d["raw"][d["cls"] == c][:, CROP:CROP + N]).items()} for c in range(3)]
    for k in real[0]:
        print(f"SIM {k}: real drone median {np.median(real[0][k]):.3f}; KS bird vs drone {ks_2samp(real[1][k], real[0][k]).statistic:.3f}")
        for f, fam in enumerate(["commercial", "stealth", "exotic"]):
            v = stats(s["sig"][s["family"] == f][:20000])[k].cpu().numpy()
            print(f"    {fam:10s} median {np.median(v):.3f}, KS vs real drone {ks_2samp(v, real[0][k]).statistic:.3f}")


for a in args:
    globals()[a]()
