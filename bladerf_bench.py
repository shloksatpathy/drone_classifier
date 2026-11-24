"""bladeRF CW bench radar -> same 1x150 Doppler input -> model (E8). NOT TESTED ON HARDWARE.

CW tone at fc + 250 kHz (IF offset), RX tuned to fc. After digital down-mixing by 250 kHz, the echo is
block-averaged over 1/17 kHz (one "sweep", as in the simulator) to a 17 kHz slow-time series.
  Mode B (time-matched):     150 samples at 17 kHz           -> 8.8 ms dwell
  Mode A (velocity-matched): every 13th 17 kHz sample, 150   -> 115 ms dwell, no anti-alias filter

Usage:
  python bladerf_bench.py capture <label> <n_dwells> [rpm]   -> captures/<label>_<time>.npz
  python bladerf_bench.py live [mode]                         (mode a or b, default a)
  python bladerf_bench.py eval captures/*.npz                 -> per-target outputs, rotor rate vs. tachometer
"""
import sys
import os
import time
import threading
import numpy as np
import torch
from config import N, RADARS, TYPES, CLASSES, wvec
from data import spectra
from model import Net, score, H3_V, H3_MD, H3_FLASH, H3_TIP, H3_RATE

FC = 5.8e9
IF = 250e3
FS = 17e3 * 120                  # 2.04 MHz: 120 raw samples per 17 kHz slow-time sample
DECIM_A = 13
TX_GAIN = 0                      # dB; set for ~0 dBm out, then sweep RX attenuation for field-level SNR
RX_GAIN = 30
RAW_PER_DWELL = {"a": N * DECIM_A * 120, "b": N * 120}


def to_slow_time(iq, mode):
    """Raw complex baseband (at FS) -> 150 slow-time samples for mode a or b."""
    t = np.arange(len(iq)) / FS
    s = (iq * np.exp(-2j * np.pi * IF * t)).reshape(-1, 120).mean(1)          # 17 kHz
    return (s[::DECIM_A] if mode == "a" else s)[:N]


class Bench:
    def __init__(self):
        from bladerf import _bladerf
        self.b = _bladerf
        self.sdr = _bladerf.BladeRF()
        self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.tx = self.sdr.Channel(_bladerf.CHANNEL_TX(0))
        for ch in (self.rx, self.tx):
            ch.frequency = FC
            ch.sample_rate = FS
            ch.bandwidth = FS
        self.rx.gain = RX_GAIN
        self.tx.gain = TX_GAIN
        for layout in (_bladerf.ChannelLayout.RX_X1, _bladerf.ChannelLayout.TX_X1):
            self.sdr.sync_config(layout=layout, fmt=_bladerf.Format.SC16_Q11, num_buffers=16,
                                 buffer_size=8192, num_transfers=8, stream_timeout=3500)
        self.rx.enable = True
        self.tx.enable = True
        # 2040 samples = exactly 250 periods of the IF tone
        tone = 0.7 * np.exp(2j * np.pi * IF * np.arange(2040) / FS)
        iq = np.empty(2 * len(tone), np.int16)
        iq[0::2], iq[1::2] = (tone.real * 2047).astype(np.int16), (tone.imag * 2047).astype(np.int16)
        self.tone = iq.tobytes()
        self.running = True
        threading.Thread(target=self._tx, daemon=True).start()

    def _tx(self):
        while self.running:
            self.sdr.sync_tx(self.tone, 2040)

    def read(self, n):
        buf = bytearray(n * 4)
        self.sdr.sync_rx(buf, n)
        iq = np.frombuffer(buf, np.int16).astype(np.float32) / 2048
        return iq[0::2] + 1j * iq[1::2]

    def close(self):
        self.running = False
        self.rx.enable = False
        self.tx.enable = False
        self.sdr.close()


def load_model():
    ck = torch.load("models/bench.pt", weights_only=False)
    m = Net(ck["size"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck["openset"]


@torch.no_grad()
def infer(m, os, sig, mode):
    radar = "bench_" + mode
    x, vm, mdm = spectra(torch.tensor(sig, dtype=torch.complex64), radar)
    o = m(x, torch.tensor(wvec(radar)).expand(len(sig), 4))
    h1, h3 = o["h1"].softmax(1).numpy(), o["h3"].numpy()
    s = score(os, o["z"].numpy(), o["h2"].numpy())
    r = RADARS[radar]
    out = []
    for k in range(len(sig)):
        c = h1[k].argmax()
        label = CLASSES[c] if c else ("unknown UAV" if s[k] >= os["tau"] else TYPES[o["h2"][k].argmax()])
        if h1[k].max() < 0.6 and s[k] >= os["tau"]:
            label = "unresolved"
        out.append(dict(label=label, p=h1[k], s_unk=s[k], v=h3[k, H3_V] * r["vmax"], mD=1 / (1 + np.exp(-h3[k, H3_MD])),
                        flash=h3[k, H3_FLASH].argmax(), tip=h3[k, H3_TIP].argmax(), rate=100 * np.exp(h3[k, H3_RATE])))
    return out


def fmt(r, mode):
    rate = f" rotor {r['rate']:.0f} Hz" if mode == "a" else ""
    return (f"{r['label']:12s} p(drone/bird/human)={np.round(r['p'], 2)} S_unk={r['s_unk']:+.2f} "
            f"v={r['v']:+.2f} m/s mD={r['mD']:.2f} flash={['0', '1', '2+'][r['flash']]} tip-bin={r['tip']}{rate}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        label, n, rpm = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else np.nan
        bench = Bench()
        raw = [bench.read(RAW_PER_DWELL["a"]) for _ in range(n)]
        bench.close()
        os.makedirs("captures", exist_ok=True)
        path = f"captures/{label}_{int(time.time())}.npz"
        np.savez(path, raw=np.array(raw, np.complex64), label=label, rpm=rpm)
        print("saved", path)
    elif cmd == "live":
        mode = sys.argv[2] if len(sys.argv) > 2 else "a"
        m, os_ = load_model()
        bench = Bench()
        try:
            while True:
                sig = to_slow_time(bench.read(RAW_PER_DWELL[mode]), mode)
                print(fmt(infer(m, os_, sig[None], mode)[0], mode), flush=True)
        finally:
            bench.close()
    elif cmd == "eval":
        m, os_ = load_model()
        for path in sys.argv[2:]:
            c = np.load(path)
            for mode in ("a", "b"):
                # a Mode A capture holds 13 consecutive Mode B dwells
                sigs = np.array([to_slow_time(r, "a")] if mode == "a" else
                                [to_slow_time(r[i * RAW_PER_DWELL["b"]:(i + 1) * RAW_PER_DWELL["b"]], "b") for i in range(DECIM_A)]
                                for r in c["raw"]).reshape(-1, N)
                res = infer(m, os_, sigs, mode)
                labels, cnt = np.unique([r["label"] for r in res], return_counts=True)
                line = f"E8 {str(c['label']):12s} mode {mode}: " + ", ".join(f"{a} {b / len(res):.0%}" for a, b in zip(labels, cnt))
                if mode == "a" and not np.isnan(c["rpm"]):
                    est = np.median([r["rate"] for r in res])
                    line += f" | rotor {est:.1f} Hz vs tachometer {c['rpm'] / 60:.1f} Hz ({est / (c['rpm'] / 60) - 1:+.1%})"
                print(line)
