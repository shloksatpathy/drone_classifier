"""Rotor / body / noise radar simulator (Chen rotating-blade model) at any wavelength and slow-time rate.

Each slow-time sample integrates over one 1/17 kHz sweep (K sub-samples), like an FMCW chirp, so short
blade flashes are not missed between samples. Sampling at the PRF aliases blade Doppler automatically.
"""
import sys
import numpy as np
from config import N, RADARS, N_SIM, TIP_BINS

K = 4                    # sub-samples per sweep
SWEEP = 1 / 17e3         # integration time per slow-time sample
FAMILIES = ["commercial", "stealth", "exotic"]
EXOTIC = ["hex", "octo", "coax", "fpv", "large", "ducted", "pusher"]


def u(rng, lo, hi, n):
    return rng.uniform(lo, hi, n)


def sample_params(family, n, rng):
    p = dict(v=u(rng, -15, 15, n), beta=np.deg2rad(u(rng, 0, 60, n)))
    sub = np.full(n, family) if family != "exotic" else rng.choice(EXOTIC, n)
    R, B, L, F, rho, snr = (np.zeros(n) for _ in range(6))
    for s in np.unique(sub):
        m = sub == s
        k = m.sum()
        r, b, l, f = {
            "commercial": (4, (2, 2), (0.10, 0.20), (60, 150)),
            "stealth": (4, (2, 3), (0.10, 0.20), (60, 150)),
            "hex": (6, (2, 3), (0.10, 0.25), (50, 120)),
            "octo": (8, (2, 3), (0.10, 0.25), (50, 120)),
            "coax": (2, (2, 2), (0.15, 0.40), (30, 100)),
            "fpv": (4, (3, 5), (0.03, 0.08), (250, 400)),
            "large": (4, (2, 2), (0.25, 0.40), (30, 70)),
            "ducted": (4, (3, 5), (0.05, 0.15), (100, 250)),
            "pusher": (1, (2, 3), (0.10, 0.20), (80, 150)),
        }[s]
        R[m] = r
        B[m] = rng.integers(b[0], b[1] + 1, k)
        L[m] = u(rng, *l, k)
        F[m] = u(rng, *f, k)
        lo_r, hi_r = {"commercial": (-15, -5), "stealth": (-40, -25), "ducted": (-35, -20)}.get(s, (-25, -5))
        lo_s, hi_s = {"commercial": (5, 30), "stealth": (-5, 10)}.get(s, (-5, 30))
        rho[m] = u(rng, lo_r, hi_r, k)
        snr[m] = u(rng, lo_s, hi_s, k)
        if s == "pusher":
            p["v"][m] = u(rng, 8, 15, k) * rng.choice([-1, 1], k)
    F = np.minimum(F, 200 / (2 * np.pi * L))              # v_tip <= 200 m/s
    p.update(R=R.astype(int), B=B.astype(int), L=L, F=F, rho=rho, snr=snr, sub=sub,
             coax=sub == "coax")
    return p


def synth(p, radar, rng):
    """Complex slow time (n, N) for the parameters p, and the number of blade flashes in the dwell."""
    lam, prf = RADARS[radar]["lam"], RADARS[radar]["prf"]
    n = len(p["v"])
    t = (np.arange(N)[:, None] / prf + np.arange(K)[None] * SWEEP / K).ravel()        # (N*K,)
    k4pi = 4 * np.pi / lam
    phase = rng.uniform(0, 2 * np.pi, (n, 8))
    direction = rng.choice([-1.0, 1.0], (n, 8))
    direction[p["coax"], 1] = -direction[p["coax"], 0]
    rotor_on = np.arange(8)[None] < p["R"][:, None]                                     # (n, 8)
    blade_on = np.arange(5)[None] < p["B"][:, None]                                     # (n, 5)
    theta = (2 * np.pi * p["F"][:, None, None, None] * direction[:, :, None, None] * t
             + phase[:, :, None, None]
             + 2 * np.pi * np.arange(5)[None, None, :, None] / p["B"][:, None, None, None])  # (n, 8, 5, T)
    # flashes of one blade (perpendicular to the line of sight, cos theta = 0) during the dwell.
    # Summed over all blades it is almost always 2+, so one blade is the informative count.
    th0 = np.mod(theta[:, 0, 0, 0] * direction[:, 0], 2 * np.pi)
    sweep = 2 * np.pi * p["F"] * RADARS[radar]["dwell"]
    flash = np.floor((th0 + sweep - np.pi / 2) / np.pi) - np.floor((th0 - np.pi / 2) / np.pi)
    a = (k4pi * p["L"] / 2 * np.cos(p["beta"]))[:, None, None, None] * np.cos(theta)
    blade = p["L"][:, None, None, None] * np.sinc(a / np.pi) * np.exp(-1j * a)
    blade = (blade * (rotor_on[:, :, None, None] & blade_on[:, None, :, None])
             * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, 8, 1, 1)))).sum((1, 2))         # (n, T)
    blade *= np.sqrt(10 ** (p["rho"] / 10) / (np.abs(blade) ** 2).mean(1))[:, None]
    body = np.exp(1j * (k4pi * p["v"][:, None] * t + rng.uniform(0, 2 * np.pi, (n, 1))))
    sig = ((body + blade * body) .reshape(n, N, K)).mean(2)
    noise_p = (np.abs(sig) ** 2).mean(1) / 10 ** (p["snr"] / 10)
    sig += np.sqrt(noise_p / 2)[:, None] * (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N)))
    return sig.astype(np.complex64), flash


def targets(p, radar, flash):
    r = RADARS[radar]
    revs = p["F"] * r["dwell"]
    tip = 2 * np.pi * p["F"] * p["L"]
    return dict(v=(p["v"] / r["vmax"]).astype(np.float32),
                flash=np.minimum(flash, 2).astype(np.int64),
                tip=np.digitize(tip, TIP_BINS).astype(np.int64),
                rate=np.log(p["F"] / 100).astype(np.float32),
                rate_ok=(revs >= 2))


def generate(radar, n=N_SIM, seed=0):
    rng = np.random.default_rng(seed)
    fam = rng.choice(3, n, p=[0.4, 0.3, 0.3])
    out = {k: [] for k in ["sig", "family", "v", "flash", "tip", "rate", "rate_ok", "sub"]}
    for i in range(0, n, 500):
        for f in range(3):
            m = fam[i:i + 500] == f
            if not m.any():
                continue
            p = sample_params(FAMILIES[f], m.sum(), rng)
            sig, flash = synth(p, radar, rng)
            out["sig"].append(sig)
            out["family"].append(np.full(m.sum(), f))
            out["sub"].append(p["sub"])
            for k, v in targets(p, radar, flash).items():
                out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


if __name__ == "__main__":
    for radar in sys.argv[1:] or RADARS:
        d = generate(radar)
        np.savez(f"dataset/sim_{radar}.npz", **d)
        print(radar, len(d["sig"]), "flash", np.bincount(d["flash"]), "tip", np.bincount(d["tip"]),
              "rate_ok", d["rate_ok"].mean())
