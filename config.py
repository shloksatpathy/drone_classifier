import numpy as np

C = 299792458.0
DATA_PATH = "dataset/data_SAAB_SIRS_77GHz_FMCW.npy"
N = 150                      # slow-time samples per dwell (model input length)
CROP = 54                    # repo's fixed 150-sweep crop starts here (54:204 of 256)
V_TIP_REF = 80.0             # m/s, reference blade-tip speed for the blade-band ratio

# Radar parameter sets: wavelength (m), slow-time rate (Hz)
RADARS = {
    "saab": dict(lam=C / 77e9, prf=17e3),
    "bench_a": dict(lam=C / 5.8e9, prf=17e3 / 13),   # velocity-matched (~1.31 kHz, every 13th 17 kHz sample)
    "bench_b": dict(lam=C / 5.8e9, prf=17e3),        # time-matched
}
for r in RADARS.values():
    r["dwell"] = N / r["prf"]
    r["vmax"] = r["lam"] * r["prf"] / 4
    r["vres"] = r["lam"] / (2 * r["dwell"])


def wvec(name):
    """Waveform vector [log lam, log PRF, log dwell, log blade-band ratio], zero for SAAB."""
    r, s = RADARS[name], RADARS["saab"]
    ratio = lambda q: 2 * V_TIP_REF / q["lam"] / q["prf"]
    return np.log([r["lam"] / s["lam"], r["prf"] / s["prf"], r["dwell"] / s["dwell"], ratio(r) / ratio(s)]).astype(np.float32)


CLASSES = ["drone", "bird", "human"]
TYPES = ["D1", "D2", "D3", "D4", "D5", "D6"]
SIZES = {"S": (16, 32, 64), "M": (24, 48, 96), "L": (32, 64, 128), "XL": (64, 128, 256)}

LOSS_W = dict(oe=0.5, phys=0.5, cons=0.2, sim_h1=0.1)   # weight of simulated drones in H1 (sweep: logs/sim_h1_sweep.txt)
EPOCHS = 15
BATCH_REAL = 256
BATCH_SIM = 128
LR = 1e-3
N_SIM = 100_000              # simulated samples per radar parameter set
TIP_BINS = [50, 100, 150]    # m/s edges -> 4 tip-speed bins
