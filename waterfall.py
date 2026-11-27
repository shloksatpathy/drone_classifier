"""Live CW Doppler waterfall for the bladeRF full-duplex bench (hand-wave tests 1-5).

TX1 sends a CW tone at fc + IF while RX1 listens at fc (bladeRF-cli, RX streamed through a FIFO).
RX is mixed down by IF and block-averaged to a 1 kHz slow-time stream. A 200 ms Hann STFT runs every 50 ms
and fills the waterfall: Doppler in Hz on the left axis, radial speed in m/s on the right (v = fd * lam / 2).

Panels
  Doppler waterfall   last 10 s, peak trace (white dots), +-1 m/s guide lines
  Echo beat           I/Q of the slow-time signal over the last 2 s: the echo beating against the leakage
  RX spectrum         raw spectrum around the LO: leakage line at +IF (set RX gain for about -10 dBFS), phase-noise skirt
  Readouts            leakage dBFS, peak Doppler and speed, level vs mark, wave period/swing, empty-room check

Usage
  python waterfall.py live [fc_hz] [rx_gain]      e.g. python waterfall.py live 5.8e9 30
  python waterfall.py replay captures/<file>.bin  (fc is read from the file name)

Keys
  r  start/stop raw IQ recording -> captures/<fc>MHz_g<gain>_<time>.bin (make the empty-room baseline first)
  s  static-line removal on/off (off shows the leakage at 0 Hz)
  f  flip the Doppler sign (set it once from test 2: hand toward = +)
  m  mark the current peak level (test 4: 1.5 m -> 2.5 m should read about -9 dB)
"""
import os
import sys
import time
import fcntl
import threading
import subprocess
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

C = 299792458.0
FPGA = "/home/shlok/Downloads/hostedxA9-latest.rbf"
FS = 1e6
BW = 0.8e6
IF = 250e3
TX_GAIN = 40
SLOW = 1000                     # slow-time rate (Hz)
D = int(FS / SLOW)              # raw samples per slow-time sample (1000 = 250 IF periods)
CHUNK = 10 * D                  # raw samples per read (10 ms)
SKIP = int(0.05 * FS)           # discard the first 50 ms (TX start-up spike)
WIN, HOP, NFFT = 200, 50, 1024  # 200 ms window (5 Hz resolution), 50 ms hop, zero-padded to ~1 Hz bins
HIST = 200                      # waterfall columns (10 s)
FDMAX = 100.0                   # Hz shown
GUARD = 3.0                     # Hz around 0 left out of the peak search
SNR_MIN = 10.0                  # dB above the column median for a peak to count
FIFO = "/tmp/bladerf_rx.fifo"
TX_FILE = "/tmp/bladerf_tone.bin"

for k in ("keymap.save", "keymap.fullscreen", "keymap.home"):
    plt.rcParams[k] = []


def start_bladerf(fc, rx_gain):
    n = 8192 * 100                                                  # whole IF periods: the TX loop is phase-continuous
    tone = 0.5 * np.exp(2j * np.pi * IF * np.arange(n) / FS)
    iq = np.empty(2 * n, np.int16)
    iq[0::2], iq[1::2] = np.round(tone.real * 2047), np.round(tone.imag * 2047)
    iq.tofile(TX_FILE)
    if not os.path.exists(FIFO):
        os.mkfifo(FIFO)
    cfg = [f"set frequency rx1 {int(fc)}", f"set frequency tx1 {int(fc)}",
           f"set samplerate rx1 {int(FS)}", f"set samplerate tx1 {int(FS)}",
           f"set bandwidth rx1 {int(BW)}", f"set bandwidth tx1 {int(BW)}",
           "set agc rx1 off", f"set gain rx1 {rx_gain}", f"set gain tx1 {TX_GAIN}",
           f"tx config file={TX_FILE} format=bin repeat=0",
           f"rx config file={FIFO} format=bin n=0",
           "tx start", "rx start", "rx wait"]
    proc = subprocess.Popen(["bladeRF-cli", "-l", FPGA] + [a for c in cfg for a in ("-e", c)])
    f = open(FIFO, "rb")
    fcntl.fcntl(f, 1031, 1 << 20)                                   # F_SETPIPE_SZ: 1 MB = 250 ms of slack while drawing
    return proc, f


class Stream:
    """Reads raw SC16 IQ in 10 ms chunks; keeps the 1 kHz slow-time series and the latest raw chunk."""

    def __init__(self, f, realtime):
        self.f, self.realtime = f, realtime
        self.lo = np.exp(-2j * np.pi * IF * np.arange(CHUNK) / FS).astype(np.complex64)
        self.slow = np.zeros(3600 * SLOW, np.complex64)              # one hour of slow time
        self.n = 0
        self.raw = np.zeros(CHUNK, np.complex64)
        self.rec = None
        self.lock = threading.Lock()
        f.read(SKIP * 4)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            b = self.f.read(CHUNK * 4)
            if len(b) < CHUNK * 4:                                  # end of a replay file
                return
            with self.lock:
                if self.rec:
                    self.rec.write(b)
            x = np.frombuffer(b, np.int16).astype(np.float32).view(np.complex64) / 2048
            self.slow[self.n:self.n + CHUNK // D] = (x * self.lo).reshape(-1, D).mean(1)
            self.raw = x
            self.n += CHUNK // D
            if self.realtime:
                time.sleep(CHUNK / FS)

    def toggle_rec(self, path):
        with self.lock:
            if self.rec:
                self.rec.close()
                self.rec = None
                return None
            self.rec = open(path, "wb")
            return path


def build(st, fc, rx_gain):
    lam = C / fc
    hz_per_ms = 2 / lam
    fb = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / SLOW))
    sel = np.abs(fb) <= FDMAX
    fd = fb[sel]
    off0 = np.abs(fd) > GUARD
    w = np.hanning(WIN).astype(np.float32)
    wn = w.sum() ** 2
    nsp = 2048
    wsp = np.hanning(nsp).astype(np.float32)
    fsp = np.fft.fftshift(np.fft.fftfreq(nsp, 1 / FS)) / 1e3

    s = dict(W=np.full((sel.sum(), HIST), np.nan), trace=np.full(HIST, np.nan), level=np.full(HIST, np.nan),
             next=WIN, static=True, sign=1, mark=np.nan, rec=None)

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[1, 1, 0.75])
    ax_w = fig.add_subplot(gs[0:2, 0])
    ax_b = fig.add_subplot(gs[2, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1:3, 1])

    t_col = (np.arange(HIST) - HIST + 1) * HOP / SLOW
    img = ax_w.imshow(s["W"], aspect="auto", origin="lower", cmap="viridis", interpolation="nearest",
                      extent=[t_col[0] - HOP / SLOW / 2, HOP / SLOW / 2, fd[0], fd[-1]])
    fig.colorbar(img, ax=ax_w, pad=0.08, label="dBFS")
    trace_ln, = ax_w.plot(t_col, s["trace"], "w.", ms=3)
    for sgn in (1, -1):
        ax_w.axhline(sgn * hz_per_ms, color="tab:red", ls="--", lw=0.8)
    ax_w.text(t_col[0], hz_per_ms + 1, " +1 m/s", color="tab:red", va="bottom")
    ax_w.text(t_col[0], -hz_per_ms - 1, " -1 m/s", color="tab:red", va="top")
    ax_w.set_xlabel("time (s)")
    ax_w.set_ylabel("Doppler (Hz)")
    ax_w.secondary_yaxis("right", functions=(lambda f: f * lam / 2, lambda v: v * 2 / lam)).set_ylabel("radial speed (m/s)")
    title = ax_w.set_title("")

    tb = np.arange(-2 * SLOW, 0) / SLOW
    i_ln, = ax_b.plot(tb, np.zeros(len(tb)), lw=0.8, label="I")
    q_ln, = ax_b.plot(tb, np.zeros(len(tb)), lw=0.8, label="Q")
    ax_b.set_xlim(tb[0], 0)
    ax_b.set_xlabel("time (s)")
    ax_b.set_title("echo beat (slow-time I/Q, last 2 s)", fontsize=10)
    ax_b.legend(loc="upper left", fontsize=8)

    sp_ln, = ax_s.plot(fsp, np.full(nsp, -150.0), lw=0.6)
    ax_s.axhline(-10, color="tab:red", ls="--", lw=0.8)
    ax_s.axvline(IF / 1e3, color="gray", ls=":", lw=0.8)
    ax_s.set_ylim(-140, 0)
    ax_s.set_xlim(fsp[0], fsp[-1])
    ax_s.set_xlabel("offset from LO (kHz)")
    ax_s.set_ylabel("dBFS")
    ax_s.set_title("RX spectrum (leakage line at +IF, target -10 dBFS)", fontsize=10)

    ax_t.axis("off")
    txt = ax_t.text(0, 1, "", va="top", family="monospace", fontsize=10, transform=ax_t.transAxes)

    def column(x):
        if s["static"]:
            x = x - x.mean()
        p = np.abs(np.fft.fftshift(np.fft.fft(x * w, NFFT))) ** 2 / wn
        c = 10 * np.log10(p[sel] + 1e-20)
        return c if s["sign"] > 0 else c[::-1]

    def update(_):
        n = st.n
        while s["next"] <= n:
            c = column(st.slow[s["next"] - WIN:s["next"]])
            k = np.argmax(np.where(off0, c, -np.inf))
            ok = c[k] - np.median(c) >= SNR_MIN
            s["W"] = np.roll(s["W"], -1, 1)
            s["W"][:, -1] = c
            s["trace"] = np.roll(s["trace"], -1)
            s["trace"][-1] = fd[k] if ok else np.nan
            s["level"] = np.roll(s["level"], -1)
            s["level"][-1] = c[k] if ok else np.nan
            s["next"] += HOP
        if n < WIN:
            return

        floor = np.nanmedian(s["W"])
        img.set_data(s["W"])
        img.set_clim(floor, floor + 50)
        trace_ln.set_ydata(s["trace"])

        b = st.slow[max(0, n - 2 * SLOW):n]
        if s["static"]:
            b = b - b.mean()
        b = np.concatenate([np.zeros(2 * SLOW - len(b), np.complex64), b])
        i_ln.set_ydata(b.real)
        q_ln.set_ydata(b.imag)
        a = 1.2 * np.abs(b).max() + 1e-12
        ax_b.set_ylim(-a, a)

        r = st.raw
        sp = np.mean(np.abs(np.fft.fft(r[:len(r) // nsp * nsp].reshape(-1, nsp) * wsp, axis=1)) ** 2, 0) / wsp.sum() ** 2
        sp_ln.set_ydata(10 * np.log10(np.fft.fftshift(sp) + 1e-20))

        leak = 20 * np.log10(np.abs(st.slow[n - WIN:n].mean()) + 1e-20)
        adc = np.abs(r).max()
        fd_now = s["trace"][-1]
        lvl = np.nanmean(s["level"][-20:]) if np.isfinite(s["level"][-20:]).any() else np.nan
        empty = np.nanmax(s["W"][off0]) - floor

        valid = np.isfinite(s["trace"])
        wave = "-\n"
        if valid.sum() > HIST // 2:
            x = np.where(valid, s["trace"], 0.0)
            x = x - x.mean()
            fr = np.fft.rfftfreq(4096, HOP / SLOW)
            band = (fr > 0.3) & (fr < 3)
            fpk = fr[band][np.argmax(np.abs(np.fft.rfft(x, 4096))[band])]
            sw = np.percentile(np.abs(s["trace"][valid]), 95)
            wave = f"period {1 / fpk:5.2f} s\n             swing  +-{sw:4.1f} Hz  (+-{sw * lam / 2:4.2f} m/s)"

        f = lambda v, fmt: format(v, fmt) if np.isfinite(v) else "-"
        txt.set_text(
            f"fc {fc / 1e9:.3f} GHz   lam {lam * 100:.2f} cm\n"
            f"{hz_per_ms:.1f} Hz per m/s   RX gain {rx_gain} dB\n"
            f"expected: 1 m/s push  +-{hz_per_ms:.1f} Hz\n"
            f"          1 Hz wave +-15 cm  +-{2 * np.pi * 0.15 * hz_per_ms:.1f} Hz\n\n"
            f"leakage line   {leak:6.1f} dBFS  (target -10)\n"
            f"ADC peak       {adc:6.2f} FS\n\n"
            f"Doppler peak   {f(fd_now, '+6.1f')} Hz  ->  {f(fd_now * lam / 2, '+5.2f')} m/s\n"
            f"peak level     {f(lvl, '6.1f')} dBFS (1 s mean)\n"
            f"delta vs mark  {f(lvl - s['mark'], '+6.1f')} dB   [m]\n\n"
            f"wave (10 s)  {wave}\n\n"
            f"empty check  max off-zero {empty:4.1f} dB above floor\n"
            f"             (peaks need >= {SNR_MIN:.0f} dB to be traced)\n\n"
            f"static removal {'ON ' if s['static'] else 'OFF'} [s]   sign {'+' if s['sign'] > 0 else '-'} [f]\n"
            f"recording      {os.path.basename(s['rec']) if s['rec'] else 'off'} [r]")
        title.set_text(f"CW Doppler waterfall  {fc / 1e9:.2f} GHz  (t = {n / SLOW:.1f} s)")

    def on_key(e):
        if e.key == "s":
            s["static"] = not s["static"]
        elif e.key == "f":
            s["sign"] = -s["sign"]
        elif e.key == "m":
            s["mark"] = np.nanmean(s["level"][-20:])
        elif e.key == "r":
            os.makedirs("captures", exist_ok=True)
            s["rec"] = st.toggle_rec(f"captures/{fc / 1e6:.0f}MHz_g{rx_gain}_{int(time.time())}.bin")
            print("recording ->", s["rec"]) if s["rec"] else print("recording stopped")

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout()
    return fig, update


if __name__ == "__main__":
    if sys.argv[1] == "live":
        fc = float(sys.argv[2]) if len(sys.argv) > 2 else 5.8e9
        gain = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        proc, f = start_bladerf(fc, gain)
        st = Stream(f, realtime=False)
    else:
        name = os.path.basename(sys.argv[2])
        fc, gain = float(name.split("MHz")[0]) * 1e6, int(name.split("_g")[1].split("_")[0])
        proc, st = None, Stream(open(sys.argv[2], "rb"), realtime=True)
    fig, update = build(st, fc, gain)
    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.show()
    if proc:
        proc.terminate()
