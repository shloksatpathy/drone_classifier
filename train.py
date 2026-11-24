import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from config import EPOCHS, BATCH_REAL, BATCH_SIM, LR, LOSS_W, N, wvec
from data import augment, spectra, mask_bands
from model import Net, fit_openset, score, H3_V, H3_MD, H3_FLASH, H3_TIP, H3_RATE

dev = "cuda" if torch.cuda.is_available() else "cpu"


def real_pool(d, idx, y1=None, y2=None):
    """Real SAAB segments with soft H1/H2 targets (one-hot from labels unless given)."""
    if y1 is None:
        y1 = np.eye(3, dtype=np.float32)[d["cls"][idx]]
        y2 = np.eye(6, dtype=np.float32)[np.maximum(d["dtype"][idx], 0)] * (d["dtype"][idx] >= 0)[:, None]
    t = lambda a: torch.tensor(a, device=dev)
    return dict(raw=t(d["raw"][idx]), x2d=t(d["x2d"][idx]), y1=t(y1), y2=t(y2), m2=t(y2.sum(1) > 0))


def sim_pool(s, radar):
    t = lambda a: torch.tensor(a, device=dev)
    p = {k: t(s[k]) for k in ["sig", "family", "v", "flash", "tip", "rate", "rate_ok"]}
    p["radar"] = radar
    return p


def wb(radar, n):
    return torch.tensor(wvec(radar), device=dev).expand(n, 4)


def circ(a):
    return torch.remainder(a + 1, 2) - 1


def inputs(model, pool, i, gen, noisy, radar="saab"):
    """Model input + label-free measurements for pool rows i."""
    sig, roll = augment(pool["raw"][i], gen, noisy)
    x, vm, mdm = spectra(sig, radar)
    if noisy:
        x = mask_bands(x, gen)
    if model.dim == 2:
        x = pool["x2d"][i]
    return x, vm, mdm


def soft_ce(logits, target, weight):
    return -(weight[target.argmax(1)] * (target * F.log_softmax(logits, 1)).sum(1)).mean()


def fit(model, real, sims, val, flags, seed=0, epochs=EPOCHS):
    """flags: set of enabled terms among {"oe", "phys", "cons", "noisy"}. sims: list of sim pools
    (empty for no simulated data). val: real pool for best-epoch selection (H1 balanced accuracy)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.to(dev)
    noisy = "noisy" in flags
    counts = real["y1"].argmax(1).bincount(minlength=3).float()
    if sims:                                    # every real batch also carries BATCH_SIM simulated drones
        counts[0] += len(real["y1"]) * BATCH_SIM / BATCH_REAL * LOSS_W["sim_h1"]
    cw = counts.sum() / (3 * counts.clamp(min=1))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state = -1, None
    n = len(real["y1"])
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        for b in range(0, n, BATCH_REAL):
            i = perm[b:b + BATCH_REAL]
            x, vm, mdm = inputs(model, real, i, gen, noisy)
            out = model(x, wb("saab", len(i)))
            loss = soft_ce(out["h1"], real["y1"][i], cw)
            m2 = real["m2"][i]
            if m2.any():
                loss = loss + soft_ce(out["h2"][m2], real["y2"][i][m2], torch.ones(6, device=dev))
            if "cons" in flags:
                loss = loss + LOSS_W["cons"] * (circ(out["h3"][:, H3_V] - vm).pow(2).mean()
                                                + (torch.sigmoid(out["h3"][:, H3_MD]) - mdm).pow(2).mean())
            for s in sims:
                j = torch.randint(0, len(s["sig"]), (BATCH_SIM // len(sims),), device=dev, generator=gen)
                sig, roll = augment(s["sig"][j], gen, noisy)
                v = circ(s["v"][j] + 2 * roll / N)
                xs, _, md = spectra(sig, s["radar"], v)
                if noisy:
                    xs = mask_bands(xs, gen)
                o = model(xs, wb(s["radar"], len(j)))
                loss = loss + LOSS_W["sim_h1"] * F.cross_entropy(o["h1"], torch.zeros_like(j), weight=cw)
                out_of_lib = s["family"][j] > 0
                if "oe" in flags and out_of_lib.any():
                    loss = loss - LOSS_W["oe"] * F.log_softmax(o["h2"][out_of_lib], 1).mean()
                if "phys" in flags:
                    h3 = o["h3"]
                    ok = s["rate_ok"][j]
                    lp = (circ(h3[:, H3_V] - v).pow(2).mean() + (torch.sigmoid(h3[:, H3_MD]) - md).pow(2).mean()
                          + F.cross_entropy(h3[:, H3_FLASH], s["flash"][j]) + F.cross_entropy(h3[:, H3_TIP], s["tip"][j]))
                    if ok.any():
                        lp = lp + (h3[ok, H3_RATE] - s["rate"][j][ok]).pow(2).mean()
                    loss = loss + LOSS_W["phys"] * lp
            opt.zero_grad()
            loss.backward()
            opt.step()
        bacc = balanced_accuracy_score(val["y1"].argmax(1).cpu(), predict(model, val)["h1"].argmax(1))
        print(f"  epoch {epoch + 1:2d} val balanced acc {bacc:.4f}", flush=True)
        if bacc > best:
            best, best_state = bacc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, pool, radar="saab"):
    """Clean-input predictions: H1 probabilities, H2 logits, H3, embedding, measurements."""
    model.eval()
    out = {k: [] for k in ["h1", "h2", "h3", "z", "vm", "mdm"]}
    key = "raw" if "raw" in pool else "sig"
    for b in range(0, len(pool[key]), 4096):
        i = torch.arange(b, min(b + 4096, len(pool[key])), device=dev)
        x, vm, mdm = inputs(model, {"raw": pool[key], "x2d": pool.get("x2d")}, i, None, False, radar)
        o = model(x, wb(radar, len(i)))
        o["h1"] = o["h1"].softmax(1)
        o.update(vm=vm, mdm=mdm)
        for k in out:
            out[k].append(o[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def openset_for(model, train, val, known):
    pt, pv = predict(model, train), predict(model, val)
    tt, tv = train["y2"].argmax(1).cpu().numpy(), val["y2"].argmax(1).cpu().numpy()
    mt, mv = train["m2"].cpu().numpy(), val["m2"].cpu().numpy()
    mt &= np.isin(tt, known)
    mv &= np.isin(tv, known)
    return fit_openset(pt["z"][mt], tt[mt], pv["z"][mv], pv["h2"][mv], known)


def cons_err(p):
    return np.abs(np.remainder(p["h3"][:, H3_V] - p["vm"] + 1, 2) - 1) + np.abs(1 / (1 + np.exp(-p["h3"][:, H3_MD])) - p["mdm"])


def noisy_student(d, split, sims, flags, sizes=("S", "M", "L"), frac=0.1, cap=20000, seed=0):
    """P1 label-scarce protocol: frac of the train split labeled, the rest unlabeled. Teacher (S,
    supervised, clean) -> students of growing size trained on labeled + filtered soft pseudo-labels with
    noise. Returns the list of models (teacher first), their open-set fits and the unknown pool."""
    rng = np.random.default_rng(seed)
    tr = np.where(split == 1)[0]
    lab = np.concatenate([rng.choice(tr[d["name"][tr] == n], max(1, int(frac * (d["name"][tr] == n).sum())), replace=False)
                          for n in np.unique(d["name"][tr])])
    unl = np.setdiff1d(tr, lab)
    L, V, U = real_pool(d, lab), real_pool(d, np.where(split == 2)[0]), real_pool(d, unl)
    print(f"P1: {len(lab)} labeled, {len(unl)} unlabeled")
    teacher = fit(Net("S"), L, sims, V, flags - {"noisy"}, seed)
    models, opensets, known = [teacher], [openset_for(teacher, L, V, range(6))], list(range(6))
    eps = np.percentile(cons_err(predict(teacher, V)), 95)
    unknown = []
    for size in sizes:
        os = opensets[-1]
        p = predict(teacher, U)
        conf, pred = p["h1"].max(1), p["h1"].argmax(1)
        s_unk = score(os, p["z"], p["h2"])
        drone = pred == 0
        keep = (conf > 0.9) & (~drone | (s_unk < os["tau"])) & (cons_err(p) < eps)
        unknown.append(unl[drone & (s_unk >= os["tau"])])
        sel = np.concatenate([rng.permutation(np.where(keep & (pred == c))[0])[:cap] for c in range(3)])
        y2 = np.exp(p["h2"][sel] - p["h2"][sel].max(1, keepdims=True))
        y2 = (y2 / y2.sum(1, keepdims=True) * drone[sel, None]).astype(np.float32)
        P = real_pool(d, unl[sel], p["h1"][sel].astype(np.float32), y2)
        both = {k: torch.cat([L[k], P[k]]) for k in L}
        acc = (pred[sel] == d["cls"][unl[sel]]).mean()
        print(f"student {size}: {len(sel)} pseudo-labels (acc {acc:.3f}), {len(unknown[-1])} to unknown pool")
        teacher = fit(Net(size), both, sims, V, flags | {"noisy"}, seed)
        models.append(teacher)
        opensets.append(openset_for(teacher, L, V, known))
    pool = np.unique(np.concatenate(unknown))
    if len(pool) >= 5:
        z = predict(teacher, real_pool(d, pool))["z"]
        c = KMeans(5, n_init=10, random_state=seed).fit_predict(z)
        print("unknown-pool clusters (true labels):")
        for k in range(5):
            names, cnt = np.unique(d["name"][pool][c == k], return_counts=True)
            print(f"  cluster {k}: " + ", ".join(f"{a} {b}" for a, b in zip(names, cnt)))
    return models, opensets, pool


if __name__ == "__main__":
    # Final models on the session split. "saab": real + SAAB sims. "bench": also bladeRF Mode A/B sims.
    import os
    from data import load_saab, session_split
    from model import Net
    os.makedirs("models", exist_ok=True)
    d = load_saab()
    sess = session_split(d["name"], d["session"])
    T, V = real_pool(d, np.where(sess == 1)[0]), real_pool(d, np.where(sess == 2)[0])
    for name, radars in [("saab", ["saab"]), ("bench", ["saab", "bench_a", "bench_b"])]:
        sims = [sim_pool(np.load(f"dataset/sim_{r}.npz"), r) for r in radars]
        m = fit(Net("S"), T, sims, V, {"oe", "phys", "cons"})
        torch.save(dict(state=m.state_dict(), size="S", openset=openset_for(m, T, V, list(range(6)))), f"models/{name}.pt")
        print("saved", f"models/{name}.pt")
