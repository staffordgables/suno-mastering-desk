#!/usr/bin/env python3
"""
suno_master.py: the Suno Mastering Desk, on your own computer.

Run it with no arguments and the desk opens in your web browser, served from
this script on 127.0.0.1. Everything runs on your own computer, and Save
writes WAV files straight to a folder on disk. The desk itself needs only Python 3.8+.

Give it audio files and it masters them in batch from the command line, using
the same chain (this mode needs numpy, scipy and soundfile):
  input trim -> low cut -> mud / harshness bands -> low shelf, presence, air
  -> top-end fizz lowpass -> mid/side width with mono bass
  -> glue compressor -> soft saturation -> 5 ms lookahead limiter

The limiter drive is fitted automatically so the master lands on a target
integrated loudness (ITU-R BS.1770 / EBU R128 gating). A loudness report for
the original and the master is printed for every file.

Requirements for batch mode:
    pip install numpy scipy soundfile
    pip install numba        # optional, makes rendering much faster

Examples:
    python suno_master.py                       # open the desk
    python suno_master.py --gui song.wav        # open the desk with a track loaded
    python suno_master.py song.wav              # batch master one file
    python suno_master.py *.wav --settings my_settings.json -o masters
    python suno_master.py --export-html desk.html   # the desk as one standalone web page
    python suno_master.py *.wav --preset acoustic --target -16
    python suno_master.py song.mp3 --set harshGain=-3.5 --set width=1.2 -o masters
    python suno_master.py song.wav --no-fit --drive 6 --bits 16
    python suno_master.py --list-presets
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy import signal
    from scipy.ndimage import minimum_filter1d
    import soundfile as sf
    HAVE_DSP = True
except ImportError as _e:
    HAVE_DSP = False
    DSP_MISSING = getattr(_e, "name", None) or str(_e)

FROZEN = getattr(sys, "frozen", False) or "__compiled__" in globals()

try:
    from numba import njit as _njit
    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        if FROZEN:
            kwargs["cache"] = False  # no writable source location inside a bundled executable
        return _njit(*args, **kwargs)
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# --------------------------------------------------------------------------
# Presets (same values as the web page)
# --------------------------------------------------------------------------

BASE = dict(
    trim=0.0,        # dB, input gain before everything
    hpf=30.0,        # Hz, low cut (10 = off)
    mudGain=-1.5,    # dB
    mudFreq=280.0,   # Hz
    harshGain=-2.0,  # dB
    harshFreq=3800.0,  # Hz
    fizz=18000.0,    # Hz, lowpass (22000 = off)
    low=1.0,         # dB, low shelf at 100 Hz
    pres=0.0,        # dB, presence at 2.8 kHz
    air=1.5,         # dB, high shelf at 12 kHz
    width=1.1,       # side gain, 1.0 = unchanged
    monoBass=120.0,  # Hz, side content below this is removed (0 = off)
    compThr=-20.0,   # dB
    compRatio=2.0,   # 1.0 = off
    compAtt=30.0,    # ms
    compRel=200.0,   # ms
    sat=0.15,        # 0..1
    ceiling=-1.0,    # dBFS
    limRel=80.0,     # ms
    target=-14.0,    # LUFS integrated
    drive=0.0,       # dB into the limiter (set by fitting)
)

PRESETS = {
    "cleanup": ("Suno cleanup", dict(BASE)),
    "loud": ("Loud and bright", {**BASE, "harshGain": -2.5, "air": 2.0, "width": 1.15, "compThr": -22.0,
                                 "compRatio": 2.5, "sat": 0.3, "limRel": 60.0, "target": -9.0}),
    "acoustic": ("Acoustic and vocal", {**BASE, "hpf": 40.0, "mudGain": -1.0, "harshGain": -3.0, "harshFreq": 4200.0,
                                        "fizz": 19000.0, "low": 0.0, "pres": 0.5, "air": 1.0, "width": 1.0,
                                        "monoBass": 100.0, "compThr": -24.0, "compRatio": 1.5, "compAtt": 40.0,
                                        "compRel": 300.0, "sat": 0.05, "limRel": 120.0, "target": -16.0}),
    "warm": ("Warm and dark", {**BASE, "harshGain": -3.5, "harshFreq": 3400.0, "fizz": 15500.0,
                               "low": 2.0, "air": 0.0, "sat": 0.3}),
    "flat": ("Limiter only", {**BASE, "hpf": 10.0, "mudGain": 0.0, "harshGain": 0.0, "fizz": 22000.0, "low": 0.0,
                              "pres": 0.0, "air": 0.0, "width": 1.0, "monoBass": 0.0, "compRatio": 1.0, "sat": 0.0}),
}


def db2lin(d):
    return 10.0 ** (d / 20.0)


def lin2db(g):
    return 20.0 * math.log10(max(g, 1e-12))


# --------------------------------------------------------------------------
# Filters (RBJ cookbook biquads)
# --------------------------------------------------------------------------

def biquad(kind, f, fs, gain_db=0.0, q=0.707):
    """Return one SOS row [b0, b1, b2, 1, a1, a2]."""
    f = min(f, fs * 0.499)
    w0 = 2 * math.pi * f / fs
    c, s = math.cos(w0), math.sin(w0)
    A = 10 ** (gain_db / 40)
    alpha = s / (2 * q)
    if kind == "lowpass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peaking":
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]; a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        sa = 2 * math.sqrt(A) * (s / 2 * math.sqrt(2))  # shelf slope S = 1
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
            a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
            a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
    else:
        raise ValueError(kind)
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]


def k_weighting(fs):
    """BS.1770 K-weighting as two SOS sections, valid at any sample rate."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / fs); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    s1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0,
          1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / fs); a0 = 1 + K / Q + K * K
    s2 = [1.0, -2.0, 1.0, 1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return np.array([s1, s2])


# --------------------------------------------------------------------------
# Recursive pieces (numba-accelerated when available)
# --------------------------------------------------------------------------

@njit(cache=True)
def env_follow(x, att, rel):
    out = np.empty_like(x)
    e = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        if v > e:
            e = att * e + (1.0 - att) * v
        else:
            e = rel * e + (1.0 - rel) * v
        out[i] = e
    return out


@njit(cache=True)
def release_smooth(held, rel):
    out = np.empty_like(held)
    g = 1.0
    for i in range(held.shape[0]):
        h = held[i]
        if h < g:
            g = h
        else:
            g = h + (g - h) * rel
        out[i] = g
    return out


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------

def lookahead_samples(fs):
    la = max(2, int(round(fs * 0.005)))
    return la + (la % 2)  # keep even so the trailing window has odd length


def process(x, fs, s):
    """x: float64 array (N, 2). Returns (y, stats)."""
    y = x * db2lin(s["trim"])

    sos = [
        biquad("highpass", max(s["hpf"], 10.0), fs, q=0.707),
        biquad("peaking", s["mudFreq"], fs, s["mudGain"], q=1.0),
        biquad("peaking", s["harshFreq"], fs, s["harshGain"], q=1.6),
        biquad("lowshelf", 100.0, fs, s["low"]),
        biquad("peaking", 2800.0, fs, s["pres"], q=0.7),
        biquad("highshelf", 12000.0, fs, s["air"]),
    ]
    if s["fizz"] < 22000 and s["fizz"] < fs * 0.49:
        sos.append(biquad("lowpass", s["fizz"], fs, q=0.707))
    y = signal.sosfilt(np.array(sos), y, axis=0)

    # Mid/side width with mono bass
    mid = 0.5 * (y[:, 0] + y[:, 1])
    side = 0.5 * (y[:, 0] - y[:, 1])
    if s["monoBass"] > 0:
        hp = biquad("highpass", s["monoBass"], fs, q=0.707)
        side = signal.sosfilt(np.array([hp, hp]), side)
    side = side * s["width"]
    y = np.stack([mid + side, mid - side], axis=1)

    stats = {"glue_gr_max_db": 0.0, "limiter_gr_max_db": 0.0}

    # Glue compressor, stereo-linked, soft knee 6 dB
    if s["compRatio"] > 1.001:
        det = np.max(np.abs(y), axis=1)
        att = math.exp(-1.0 / (max(0.0005, s["compAtt"] / 1000) * fs))
        rel = math.exp(-1.0 / (max(0.005, s["compRel"] / 1000) * fs))
        env = env_follow(det, att, rel)
        over = 20 * np.log10(env + 1e-9) - s["compThr"]
        W, slope = 6.0, 1.0 / s["compRatio"] - 1.0
        gr = np.where(2 * over <= -W, 0.0,
                      np.where(2 * np.abs(over) < W, slope * (over + W / 2) ** 2 / (2 * W), slope * over))
        gr = np.minimum(gr, 0.0)
        y = y * db2lin(gr)[:, None]
        stats["glue_gr_max_db"] = max(0.0, float(-gr.min()))

    # Soft saturation (unity gain for small signals)
    a = s["sat"]
    if a > 0.001:
        k = 1 + 2 * a
        y = (1 - a) * y + a * np.tanh(k * y) / k

    # Lookahead brickwall limiter
    y = y * db2lin(s["drive"])
    c = db2lin(s["ceiling"])
    la = lookahead_samples(fs)
    n = y.shape[0]
    yp = np.concatenate([y, np.zeros((la, 2))], axis=0)
    pk = np.max(np.abs(yp), axis=1)
    req = np.where(pk > c, c / np.maximum(pk, 1e-12), 1.0)
    w = la + 1
    held = minimum_filter1d(req, w, origin=(w - 1) // 2, mode="nearest")  # min over n-la..n
    g = release_smooth(held, math.exp(-1.0 / (max(0.005, s["limRel"] / 1000) * fs)))
    # history before the start counts as gain 1
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([np.ones(la), g]))])
    sm = (cs[la + 1:] - cs[1:-la]) / la  # trailing mean over la samples, aligned to g
    sm = np.minimum(sm, 1.0)
    out = yp[:n] * sm[la:la + n, None]
    out = np.clip(out, -c, c)
    stats["limiter_gr_max_db"] = max(0.0, float(-lin2db(sm[la:la + n].min())))
    return out, stats


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def true_peak(x, chunk=1 << 20):
    m = float(np.max(np.abs(x)))
    pad = 64
    for start in range(0, x.shape[0], chunk):
        a = max(0, start - pad)
        b = min(x.shape[0], start + chunk + pad)
        up = signal.resample_poly(x[a:b], 4, 1, axis=0)
        lo = (start - a) * 4
        hi = lo + (min(start + chunk, x.shape[0]) - start) * 4
        m = max(m, float(np.max(np.abs(up[lo:hi]))))
    return lin2db(m)


def analyze(x, fs, tp=True):
    kw = signal.sosfilt(k_weighting(fs), x, axis=0)
    seg = int(round(fs * 0.1))
    nseg = x.shape[0] // seg
    L = lambda e: -0.691 + 10 * np.log10(np.asarray(e) + 1e-20)
    res = {"integrated_lufs": -math.inf, "short_term_max_lufs": -math.inf, "loudness_range_lu": float("nan"),
           "sample_peak_dbfs": lin2db(float(np.max(np.abs(x)))), "true_peak_dbtp": float("nan")}
    if nseg >= 4:
        E = (kw[:nseg * seg] ** 2).reshape(nseg, seg, -1).mean(axis=1).sum(axis=1)
        P = np.concatenate([[0.0], np.cumsum(E)])
        blk = (P[4:] - P[:-4]) / 4
        blk = blk[L(blk) > -70]
        if blk.size:
            rel = L(blk.mean()) - 10
            g = blk[L(blk) > rel]
            if g.size:
                res["integrated_lufs"] = float(L(g.mean()))
        if nseg >= 30:
            st = (P[30:] - P[:-30]) / 30
            res["short_term_max_lufs"] = float(L(st).max())
            st = st[L(st) > -70]
            if st.size > 2:
                rel = L(st.mean()) - 20
                v = np.sort(L(st[L(st) > rel]))
                if v.size > 1:
                    res["loudness_range_lu"] = float(v[int(round(0.95 * (v.size - 1)))] - v[int(round(0.1 * (v.size - 1)))])
    if tp:
        res["true_peak_dbtp"] = true_peak(x)
    return res


# --------------------------------------------------------------------------
# Fitting, I/O, report
# --------------------------------------------------------------------------

def fit(x, fs, s, verbose=True):
    s = dict(s)
    d, prev = s["drive"], None
    y, st, an = None, None, None
    for i in range(6):
        s["drive"] = d
        y, st = process(x, fs, s)
        an = analyze(y, fs, tp=False)
        I = an["integrated_lufs"]
        if verbose:
            print(f"    pass {i + 1}: drive {d:+.2f} dB -> {I:.2f} LUFS")
        if not math.isfinite(I):
            break
        err = s["target"] - I
        if abs(err) < 0.1:
            break
        slope = 1.0
        if prev and abs(d - prev[0]) > 1e-3:
            sl = (I - prev[1]) / (d - prev[0])
            if math.isfinite(sl):
                slope = min(1.2, max(0.15, sl))
        prev = (d, I)
        nd = min(24.0, max(-12.0, d + err / slope))
        if abs(nd - d) < 0.01:
            break
        d = nd
    return y, st, s


def load(path):
    try:
        x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as e:
        sys.exit(f"Could not read {path}: {e}\n"
                 "MP3 needs libsndfile 1.1 or newer; otherwise convert to WAV first, for example with ffmpeg.")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x, fs


def save(path, y, fs, bits, fmt):
    if bits == 16:
        y = y + (np.random.random_sample(y.shape) - np.random.random_sample(y.shape)) / 32768.0  # TPDF dither
        subtype = "PCM_16"
    else:
        subtype = "PCM_24"
    sf.write(str(path), np.clip(y, -1.0, 1.0), fs, subtype=subtype, format=fmt.upper())


def fmt_v(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if not math.isfinite(v):
        return "-inf"
    return f"{v:6.1f}"


def report(o, m, s):
    rows = [("Integrated (LUFS)", "integrated_lufs"), ("Short-term max (LUFS)", "short_term_max_lufs"),
            ("True peak (dBTP)", "true_peak_dbtp"), ("Sample peak (dBFS)", "sample_peak_dbfs"),
            ("Loudness range (LU)", "loudness_range_lu")]
    print(f"    {'':24}{'Original':>10}{'Master':>10}")
    for label, k in rows:
        print(f"    {label:24}{fmt_v(o[k]):>10}{fmt_v(m[k]):>10}")
    plr_o = o["true_peak_dbtp"] - o["integrated_lufs"]
    plr_m = m["true_peak_dbtp"] - m["integrated_lufs"]
    print(f"    {'Peak to loudness (dB)':24}{fmt_v(plr_o):>10}{fmt_v(plr_m):>10}")
    notes = []
    if o["true_peak_dbtp"] > -0.2:
        notes.append(f"The original reaches {o['true_peak_dbtp']:.1f} dBTP, which suggests it was already limited "
                     "or clipped on export. Extra limiting will add little loudness.")
    if o["integrated_lufs"] > -11:
        notes.append(f"The original already sits at {o['integrated_lufs']:.1f} LUFS. Streaming services turn louder "
                     "tracks down to about -14 LUFS, so a more open master may sound better after normalization.")
    if m["true_peak_dbtp"] > s["ceiling"] + 0.3:
        notes.append(f"Inter-sample peaks reach {m['true_peak_dbtp']:.1f} dBTP. Lower --ceiling a few tenths "
                     "if a distributor asks for -1 dBTP.")
    if math.isfinite(m["loudness_range_lu"]) and m["loudness_range_lu"] < 3:
        notes.append(f"Loudness range is narrow at {m['loudness_range_lu']:.1f} LU. Ease the glue compressor or "
                     "the target if the track feels flat.")
    for n in notes:
        print("    note: " + n)
    return notes


def parse_set(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        if k not in BASE:
            sys.exit(f"Unknown setting {k!r}. Valid keys: {', '.join(BASE)}")
        out[k] = float(v)
    return out


# --------------------------------------------------------------------------
# The desk: a local web server for the browser interface
# --------------------------------------------------------------------------

def unique_path(folder, name):
    p = folder / name
    k = 2
    while p.exists():
        p = folder / f"{Path(name).stem} ({k}){Path(name).suffix}"
        k += 1
    return p


def page_html(cfg=None):
    """The desk as a complete HTML document. cfg switches on local save mode."""
    boot = ""
    if cfg is not None:
        boot = "<script>window.__SUNO_LOCAL__=" + json.dumps(cfg).replace("<", "\\u003c") + ";</script>"
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{margin:0}[hidden]{display:none!important}</style>'
            + boot + '</head><body>' + PAGE_HTML + '</body></html>')


def run_gui(initial, outdir, port, open_browser):
    import secrets
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs

    if initial is not None and not initial.is_file():
        sys.exit(f"Not a file: {initial}")
    outdir = outdir.expanduser().resolve()
    token = secrets.token_urlsafe(24)
    server = None
    MAX_BODY = 4 << 30

    class Desk(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="text/plain; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            # Refuse requests for any other host name, which blocks DNS-rebinding pages
            port_ = server.server_address[1]
            return self.headers.get("Host", "") in (f"127.0.0.1:{port_}", f"localhost:{port_}")

        def _token_ok(self):
            return secrets.compare_digest(self.headers.get("X-Desk-Token", ""), token)

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, "Forbidden")
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                cfg = {"token": token, "outdir": str(outdir), "initial": initial.name if initial else None}
                return self._send(200, page_html(cfg), "text/html; charset=utf-8")
            if path == "/api/initial" and initial and self._token_ok():
                return self._send(200, initial.read_bytes(), "application/octet-stream")
            if path == "/favicon.ico":
                return self._send(204, b"")
            return self._send(404, "Not found")

        def do_POST(self):
            if not (self._host_ok() and self._token_ok()):
                return self._send(403, "Forbidden")
            url = urlparse(self.path)
            if url.path != "/api/save":
                return self._send(404, "Not found")
            raw = parse_qs(url.query).get("name", [""])[0]
            name = re.sub(r"[^\w\- .()]+", "", Path(raw).name).strip(" .")
            if not name or Path(name).suffix.lower() not in (".wav", ".json"):
                return self._send(400, "Only .wav and .json files can be saved.")
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._send(400, "Empty or oversized file.")
            try:
                outdir.mkdir(parents=True, exist_ok=True)
                dest = unique_path(outdir, name)
                with open(dest, "xb") as fh:
                    left = length
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            raise IOError("upload ended early")
                        fh.write(chunk)
                        left -= len(chunk)
            except Exception as e:
                return self._send(500, str(e))
            print(f"  saved {dest}")
            return self._send(200, json.dumps({"path": str(dest)}), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Desk)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Suno Mastering Desk is running at {url}")
    print(f"Masters and settings save to {outdir}")
    if initial:
        print(f"Loading {initial}")
    print("Press Ctrl+C here to stop it.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main():
    ap = argparse.ArgumentParser(description="Master Suno exports to a loudness target.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Examples:")[1])
    ap.add_argument("inputs", nargs="*", type=Path, help="audio files (WAV, FLAC, MP3); none opens the desk")
    ap.add_argument("--gui", action="store_true", help="open the desk, loading the first input if one is given")
    ap.add_argument("--port", type=int, default=0, help="port for the desk (default: any free port)")
    ap.add_argument("--no-browser", action="store_true", help="start the desk without opening a browser")
    ap.add_argument("-o", "--outdir", type=Path, help="output folder (batch default: next to each input; "
                    "desk default: next to the loaded track, else ~/Music/Suno Masters)")
    ap.add_argument("--settings", type=Path, help="settings JSON saved from the desk")
    ap.add_argument("-p", "--preset", default="cleanup", choices=PRESETS)
    ap.add_argument("-t", "--target", type=float, help="integrated loudness target in LUFS")
    ap.add_argument("-c", "--ceiling", type=float, help="limiter ceiling in dBFS")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any setting, repeatable")
    ap.add_argument("--no-fit", action="store_true", help="skip fitting and use --drive as given")
    ap.add_argument("--drive", type=float, help="limiter drive in dB (start point for fitting, or fixed with --no-fit)")
    ap.add_argument("--bits", type=int, choices=(16, 24), default=24)
    ap.add_argument("--format", choices=("wav", "flac"), default="wav")
    ap.add_argument("--suffix", default="_master")
    ap.add_argument("--json", type=Path, help="write all reports and settings to this JSON file")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--export-html", type=Path, metavar="FILE", help="write the desk as a standalone web page and exit")
    args = ap.parse_args()

    if args.list_presets:
        for k, (name, v) in PRESETS.items():
            diff = {kk: vv for kk, vv in v.items() if BASE[kk] != vv}
            print(f"{k:10} {name:20} target {v['target']:.0f} LUFS  {diff if diff else ''}")
        return
    if args.export_html:
        args.export_html.write_text(page_html(), encoding="utf-8")
        print(f"Wrote {args.export_html}")
        return
    if args.gui or not args.inputs:
        initial = args.inputs[0].expanduser().resolve() if args.inputs else None
        outdir = args.outdir or (initial.parent if initial else Path.home() / "Music" / "Suno Masters")
        run_gui(initial, outdir, args.port, not args.no_browser)
        return
    if not HAVE_DSP:
        sys.exit(f"Batch mode needs numpy, scipy and soundfile (missing: {DSP_MISSING}).\n"
                 "Install them with: pip install numpy scipy soundfile\n"
                 "The desk works without them: run this script with no arguments.")
    if not HAVE_NUMBA:
        print("numba is not installed, so rendering runs in plain Python and will be slow. pip install numba to speed it up.\n")

    base = dict(PRESETS[args.preset][1])
    if args.settings:
        try:
            loaded = json.loads(args.settings.read_text())
        except Exception as e:
            sys.exit(f"Could not read settings from {args.settings}: {e}")
        if loaded.get("preset") in PRESETS and "-p" not in sys.argv and "--preset" not in sys.argv:
            base = dict(PRESETS[loaded["preset"]][1])
        base.update({k: float(v) for k, v in loaded.items() if k in BASE and isinstance(v, (int, float))})
    if args.target is not None:
        base["target"] = args.target
    if args.ceiling is not None:
        base["ceiling"] = args.ceiling
    base.update(parse_set(args.set))

    all_reports = []
    for path in args.inputs:
        print(f"{path.name}")
        x, fs = load(path)
        dur = x.shape[0] / fs
        print(f"    {int(dur // 60)}:{dur % 60:04.1f}  {fs / 1000:.1f} kHz")
        orig = analyze(x, fs)
        s = dict(base)
        if args.drive is not None:
            s["drive"] = args.drive
        elif math.isfinite(orig["integrated_lufs"]):
            s["drive"] = max(-12.0, min(24.0, s["target"] - orig["integrated_lufs"]))
        if args.no_fit:
            y, st = process(x, fs, s)
        else:
            print(f"    fitting to {s['target']:.1f} LUFS")
            y, st, s = fit(x, fs, s)
        m = analyze(y, fs)
        print(f"    drive {s['drive']:+.2f} dB, glue GR max {st['glue_gr_max_db']:.1f} dB, "
              f"limiter GR max {st['limiter_gr_max_db']:.1f} dB")
        notes = report(orig, m, s)
        outdir = args.outdir or path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{path.stem}{args.suffix}.{args.format}"
        save(out, y, fs, args.bits, args.format)
        print(f"    wrote {out}\n")
        all_reports.append({"input": str(path), "output": str(out), "sample_rate": fs, "settings": s,
                            "original": orig, "master": m, "dynamics": st, "notes": notes})

    if args.json:
        args.json.write_text(json.dumps(all_reports, indent=2, default=lambda v: None))
        print(f"Report written to {args.json}")


PAGE_HTML = r"""
<title>Suno Mastering Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: hardware rack. Deck screen on top, control modules left, meters and report right; stacks to one column on phones. */
:root{
  --bg:#E2E5E8; --panel:#F3F4F5; --panel-2:#E8EBEE; --ink:#1B2028; --ink-2:#596270; --line:#C6CCD3;
  --accent:#A55807; --accent-strong:#8A4904; --on-accent:#FFFFFF;
  --screen:#161A20; --screen-grid:#29303A; --screen-ink:#94A0AE; --trace:#F2A541; --trace-pre:#6E7987;
  --good:#2E8B57; --warn:#B7860B; --crit:#C0392B;
  --cap:#FBFBFC; --cap-edge:#A3ABB5;
  --f-disp:"Barlow Condensed","Arial Narrow",sans-serif;
  --f-body:"Barlow",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#111418; --panel:#191D23; --panel-2:#1F242B; --ink:#E2E6EB; --ink-2:#97A0AB; --line:#2F363F;
    --accent:#F0A23B; --accent-strong:#F6B35C; --on-accent:#1A1104;
    --screen:#0B0E12; --screen-grid:#222930; --screen-ink:#8D97A3; --trace:#F2A541; --trace-pre:#5F6A77;
    --good:#45B97C; --warn:#E0B13F; --crit:#E8584C; --cap:#2B323B; --cap-edge:#4C5560;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#111418; --panel:#191D23; --panel-2:#1F242B; --ink:#E2E6EB; --ink-2:#97A0AB; --line:#2F363F;
  --accent:#F0A23B; --accent-strong:#F6B35C; --on-accent:#1A1104;
  --screen:#0B0E12; --screen-grid:#222930; --screen-ink:#8D97A3; --trace:#F2A541; --trace-pre:#5F6A77;
  --good:#45B97C; --warn:#E0B13F; --crit:#E8584C; --cap:#2B323B; --cap-edge:#4C5560;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.45;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 48px}
.wrap{max-width:1280px;margin:0 auto;display:grid;gap:16px}
.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
h1,h2{font-family:var(--f-disp);margin:0;text-wrap:balance}
h2{font-size:13px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-2)}
button,select,input{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* header */
.top{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px 24px}
.brand{display:flex;align-items:baseline;gap:10px;margin-right:auto}
.brand .eyebrow{font-family:var(--f-disp);font-weight:700;font-size:14px;letter-spacing:.2em;text-transform:uppercase;color:var(--on-accent);background:var(--accent);padding:2px 8px;border-radius:3px}
.brand h1{font-size:34px;font-weight:600;letter-spacing:.02em;line-height:1}
.load{display:flex;gap:8px;flex-wrap:wrap}
.preset{display:flex;align-items:center;gap:8px}
.preset label{font-family:var(--f-disp);font-size:13px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-2)}
select,input[type=text]{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:7px 10px;min-width:0}

.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;border:1px solid var(--line);background:var(--panel);
  border-radius:4px;padding:7px 14px;font-weight:600;cursor:pointer;line-height:1.2;white-space:nowrap}
.btn:hover{border-color:var(--ink-2)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.btn.primary:hover{background:var(--accent-strong);border-color:var(--accent-strong)}
.btn:disabled{opacity:.5;cursor:default}

.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;display:grid;gap:12px;min-width:0}

/* deck */
.trackline{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 16px;align-items:baseline}
#trackName{font-weight:600;font-size:16px;min-width:0;overflow-wrap:anywhere}
#trackMeta{color:var(--ink-2);font-size:12px}
.screen{background:var(--screen);border-radius:4px;position:relative;overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,.45)}
.wave{height:120px;cursor:pointer}
.screen canvas{display:block;width:100%;height:100%}
.drophint{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--screen-ink);font-family:var(--f-disp);
  letter-spacing:.1em;text-transform:uppercase;font-size:14px;pointer-events:none;background:rgba(11,14,18,.82)}
body.dragging .drophint{display:flex!important}
.transport{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px}
.play{min-width:84px}
.time{font-size:13px;color:var(--ink-2)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:4px;overflow:hidden}
.seg button{border:0;background:var(--panel);padding:7px 12px;cursor:pointer;font-weight:600}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--panel)}
.chk{display:inline-flex;align-items:center;gap:6px;font-size:14px;color:var(--ink-2);cursor:pointer}
.chk input{accent-color:var(--accent)}
.status{margin:0;font-size:13px;color:var(--ink-2);min-height:1.45em}
.status.busy::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:8px;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.status.busy::before{animation:none}}

/* grid */
.grid{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;align-items:start}
@media (max-width:980px){.grid{grid-template-columns:minmax(0,1fr)}}
.rack{border:0;margin:0;padding:0;min-width:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px;align-items:start}
.rack:disabled{opacity:.6}
.side{display:grid;gap:16px;min-width:0}
.mod .note{margin:0;font-size:13px;color:var(--ink-2);max-width:60ch}
.modhead{display:flex;justify-content:space-between;align-items:center;gap:8px}
.modhead .idx{font-family:var(--f-mono);font-size:11px;color:var(--ink-2)}
.ctl{display:grid;grid-template-columns:1fr auto;align-items:center;column-gap:8px}
.ctl label{font-size:14px;font-weight:500}
.ctl output{font-family:var(--f-mono);font-size:12px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.ctl input{grid-column:1/-1}

/* help */
.ht{display:inline-flex;align-items:center;gap:8px;cursor:help}
.help{width:20px;height:20px;border-radius:50%;border:1px solid var(--line);background:var(--panel-2);color:var(--ink-2);
  font:600 12px/1 var(--f-body);display:inline-flex;align-items:center;justify-content:center;padding:0;flex:none;cursor:help}
.ht:hover .help,.help[aria-expanded="true"]{border-color:var(--accent);color:var(--accent)}
[data-tip]{text-decoration:underline dotted var(--ink-2);text-decoration-thickness:1px;text-underline-offset:4px;cursor:help}
#tip{position:fixed;z-index:50;left:0;top:0;width:max-content;max-width:min(340px,calc(100vw - 24px));background:var(--ink);color:var(--panel);
  padding:10px 12px;border-radius:6px;font-size:13px;line-height:1.45;box-shadow:0 6px 20px rgba(0,0,0,.28);pointer-events:none}
#tip p{margin:0}
#tip p+p{margin-top:6px}
#tip strong{display:block;font-family:var(--f-disp);font-weight:600;letter-spacing:.1em;text-transform:uppercase;font-size:12px;opacity:.7;margin-bottom:4px}

/* fader */
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:30px;background:transparent;margin:0;cursor:pointer;
  --a:0%;--b:50%}
input[type=range]::-webkit-slider-runnable-track{height:4px;border-radius:2px;
  background:linear-gradient(to right,transparent var(--a),var(--accent) var(--a),var(--accent) var(--b),transparent var(--b)),var(--line)}
input[type=range]::-moz-range-track{height:4px;border-radius:2px;
  background:linear-gradient(to right,transparent var(--a),var(--accent) var(--a),var(--accent) var(--b),transparent var(--b)),var(--line)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:24px;margin-top:-10px;border-radius:3px;
  background:linear-gradient(to right,transparent 6px,var(--ink) 6px,var(--ink) 8px,transparent 8px),var(--cap);border:1px solid var(--cap-edge);box-shadow:0 1px 2px rgba(0,0,0,.25)}
input[type=range]::-moz-range-thumb{width:12px;height:22px;border-radius:3px;
  background:linear-gradient(to right,transparent 5px,var(--ink) 5px,var(--ink) 7px,transparent 7px),var(--cap);border:1px solid var(--cap-edge);box-shadow:0 1px 2px rgba(0,0,0,.25)}

.targets{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
.targets button{border:1px solid var(--line);background:var(--panel-2);border-radius:4px;padding:6px 8px;text-align:left;cursor:pointer;display:grid;gap:0}
.targets button b{font-family:var(--f-mono);font-size:14px;font-weight:500}
.targets button span{font-size:12px;color:var(--ink-2)}
.targets button[aria-pressed="true"]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}

/* spectrum */
.spec{height:190px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink-2);flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}

/* meters */
.mrow{display:grid;grid-template-columns:88px minmax(0,1fr) 64px;align-items:center;gap:10px}
.ml{font-size:13px;color:var(--ink-2)}
.mv{font-size:13px;text-align:right}
.mbar{height:10px;background:var(--panel-2);border:1px solid var(--line);border-radius:2px;position:relative;overflow:hidden}
.mbar i{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--accent)}
.mbar.gr i{background:var(--warn)}
.mbar.hot i{background:var(--crit)}
.mbar b{position:absolute;top:-1px;bottom:-1px;width:2px;background:var(--ink);opacity:.7}
.scale{display:flex;justify-content:space-between;font-size:10px;color:var(--ink-2);margin-left:98px;margin-right:74px}

/* report */
table.report{width:100%;border-collapse:collapse;font-size:13px}
.report th,.report td{padding:6px 4px;border-bottom:1px solid var(--line);text-align:right}
.report th:first-child,.report td:first-child{text-align:left;color:var(--ink-2);font-weight:500}
.report thead th{font-family:var(--f-disp);letter-spacing:.1em;text-transform:uppercase;font-size:12px;color:var(--ink-2);font-weight:600}
.report td{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.report.stale td:nth-child(3){opacity:.45}
.notes{display:grid;gap:6px;margin:0;padding:0;list-style:none}
.notes li{font-size:13px;padding-left:12px;border-left:3px solid var(--warn)}
.stale-flag{margin:0;font-size:12px;color:var(--warn);font-weight:600}
.actions{display:flex;gap:8px;flex-wrap:wrap}
.exp{display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px 10px;align-items:center}
.exp label{font-size:13px;color:var(--ink-2)}
.exp .btn{grid-column:1/-1}
.exp .actions{grid-column:1/-1}
.exp .actions .btn{grid-column:auto}
.fine{grid-column:1/-1;margin:0;font-size:12px;color:var(--ink-2)}
</style>

<div class="wrap">
  <header class="top">
    <div class="brand"><span class="eyebrow">Suno</span><h1>Mastering Desk</h1></div>
    <div class="load">
      <label class="btn primary" for="file" tabindex="0" id="loadLbl">Load track</label>
      <input id="file" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg" hidden>
      <button class="btn" id="demoBtn" type="button">Demo loop</button>
    </div>
    <div class="preset"><label for="preset" data-tip="preset">Preset</label><select id="preset"></select></div>
  </header>

  <section class="panel deck" aria-label="Track" data-help="deck">
    <div class="modhead"><h2>Listen and compare</h2></div>
    <div class="trackline"><span id="trackName">Preparing demo loop</span><span id="trackMeta" class="mono"></span></div>
    <div class="screen wave" id="waveBox"><canvas id="wave" aria-label="Waveform. Click to move the playhead."></canvas><div class="drophint" id="dropHint" hidden>Drop a WAV or MP3 exported from Suno</div></div>
    <div class="transport">
      <button class="btn primary play" id="playBtn" type="button">Play</button>
      <span class="mono time" id="time">0:00 / 0:00</span>
      <div class="seg" role="group" aria-label="Listen to">
        <button id="abOrig" type="button" aria-pressed="false">Original</button><button id="abMaster" type="button" aria-pressed="true">Master</button>
      </div>
      <label class="chk"><input type="checkbox" id="match" checked> Level-matched A/B</label>
      <label class="chk"><input type="checkbox" id="loop" checked> Loop</label>
    </div>
    <p class="status" id="status" role="status" aria-live="polite"></p>
  </section>

  <div class="grid">
    <fieldset class="rack" id="rack"></fieldset>
    <aside class="side">
      <section class="panel" data-help="spectrum">
        <div class="modhead"><h2>Spectrum</h2></div>
        <div class="screen spec"><canvas id="spec" aria-label="Live spectrum of original and output"></canvas></div>
        <div class="legend"><span><i style="background:var(--trace-pre)"></i>Original</span><span><i style="background:var(--trace)"></i>What you hear</span><span>Tilted 3 dB/oct around 1 kHz</span></div>
      </section>
      <section class="panel" data-help="meters">
        <div class="modhead"><h2>Output meters</h2></div>
        <div class="mrow"><span class="ml">Momentary</span><div class="mbar"><i id="mb-m"></i><b id="tg-m"></b></div><span class="mv mono" id="mv-m"></span></div>
        <div class="mrow"><span class="ml">Short-term</span><div class="mbar"><i id="mb-s"></i><b id="tg-s"></b></div><span class="mv mono" id="mv-s"></span></div>
        <div class="mrow"><span class="ml">Peak</span><div class="mbar" id="mbox-p"><i id="mb-p"></i></div><span class="mv mono" id="mv-p"></span></div>
        <div class="scale mono"><span>&minus;40</span><span>&minus;20</span><span>0</span></div>
        <div class="mrow"><span class="ml">Glue GR</span><div class="mbar gr"><i id="mb-c"></i></div><span class="mv mono" id="mv-c"></span></div>
        <div class="mrow"><span class="ml">Limiter GR</span><div class="mbar gr"><i id="mb-l"></i></div><span class="mv mono" id="mv-l"></span></div>
        <div class="scale mono"><span>0</span><span>6</span><span>12 dB</span></div>
      </section>
      <section class="panel" data-help="report">
        <div class="modhead"><h2>Loudness report</h2></div>
        <table class="report" id="report"><thead><tr><th></th><th>Original</th><th>Master</th></tr></thead><tbody id="repBody"></tbody></table>
        <p class="stale-flag" id="staleFlag" hidden>Settings changed since the master was measured.</p>
        <ul class="notes" id="notes"></ul>
        <div class="actions">
          <button class="btn primary" id="fitBtn" type="button">Fit to target</button>
          <button class="btn" id="measureBtn" type="button">Measure as set</button>
        </div>
      </section>
      <section class="panel" data-help="export">
        <div class="modhead"><h2>Export</h2></div>
        <div class="exp">
          <label for="fname">File name</label><input id="fname" type="text" value="demo-loop_master" autocomplete="off">
          <label for="bits">Format</label>
          <select id="bits"><option value="24">WAV, 24-bit</option><option value="16">WAV, 16-bit with dither</option></select>
          <button class="btn primary" id="saveBtn" type="button">Save master</button>
          <div class="actions setrow">
            <button class="btn" id="saveSet" type="button">Save settings</button>
            <label class="btn" for="setFile" tabindex="0" id="setLbl">Load settings</label>
            <input id="setFile" type="file" accept=".json,application/json" hidden>
          </div>
          <p class="fine" id="saveNote">Saves a WAV at the track's own sample rate. Settings that changed since the last measurement are rendered first.</p>
          <p class="fine">Settings files also drive the Python batch script: <span class="mono">--settings name.json</span></p>
        </div>
      </section>
    </aside>
  </div>
</div>

<div id="tip" role="tooltip" hidden></div>
<script>
(() => {
'use strict';
const $ = s => document.querySelector(s);
const MINUS = '−';
const db2lin = d => Math.pow(10, d / 20);
const lin2db = g => 20 * Math.log10(Math.max(g, 1e-12));
const sgn = (v, d = 1) => (v < -0.0001 ? MINUS : v > 0.0001 ? '+' : '') + Math.abs(v).toFixed(d);
const num = (v, d = 1) => !isFinite(v) ? MINUS + '∞' : (v < 0 ? MINUS : '') + Math.abs(v).toFixed(d);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const BUTTER_DB = 20 * Math.log10(Math.SQRT1_2);
const tick = () => new Promise(r => setTimeout(r, 16));
const fmtTime = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + String(s).padStart(2, '0'); };

/* ---------- BS.1770 K-weighting ---------- */
function kCoefs(fs) {
  let f0 = 1681.974450955533, G = 3.999843853973347, Q = 0.7071752369554196;
  let K = Math.tan(Math.PI * f0 / fs), Vh = Math.pow(10, G / 20), Vb = Math.pow(Vh, 0.4996667741545416), a0 = 1 + K / Q + K * K;
  const s1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0];
  f0 = 38.13547087602444; Q = 0.5003270373238773; K = Math.tan(Math.PI * f0 / fs); a0 = 1 + K / Q + K * K;
  const s2 = [1, -2, 1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0];
  return [s1, s2];
}

/* ---------- Audio worklets ---------- */
const WORKLET = kCoefs.toString() + `
const DEF={compThr:-18,compRatio:2,compAtt:0.03,compRel:0.2,knee:6,sat:0.15,drive:0,ceiling:-1,limRel:0.08};
class MasterDyn extends AudioWorkletProcessor{
  constructor(o){super();const po=(o&&o.processorOptions)||{};
    this.p=Object.assign({},DEF,po.params||{});this.report=!!po.report;
    const N=this.la=Math.max(1,Math.round(sampleRate*0.005));
    this.dL=new Float32Array(N);this.dR=new Float32Array(N);this.di=0;
    this.gb=new Float64Array(N).fill(1);this.gi=0;this.gsum=N;
    this.cap=N+2;this.qv=new Float64Array(this.cap);this.qp=new Float64Array(this.cap);this.qh=0;this.qn=0;
    this.n=0;this.env=0;this.g=1;this.minC=1;this.minL=1;this.cnt=0;this.repN=Math.round(sampleRate*0.05);
    this.setup();this.port.onmessage=e=>{Object.assign(this.p,e.data);this.setup();};}
  setup(){const p=this.p,sr=sampleRate;
    this.att=Math.exp(-1/(Math.max(0.0005,p.compAtt)*sr));this.rel=Math.exp(-1/(Math.max(0.005,p.compRel)*sr));
    this.lrel=Math.exp(-1/(Math.max(0.005,p.limRel)*sr));this.drv=Math.pow(10,p.drive/20);this.ceil=Math.pow(10,p.ceiling/20);this.k=1+2*p.sat;}
  process(inputs,outputs){
    const out=outputs[0],oL=out[0],oR=out[1]||null;const inp=inputs[0];
    const iL=inp&&inp[0],iR=inp&&(inp[1]||inp[0]);const n=oL.length,p=this.p,la=this.la,cap=this.cap;
    const thr=p.compThr,slope=1/p.compRatio-1,W=p.knee,compOn=p.compRatio>1.001,sat=p.sat,k=this.k,tk=1/k,c=this.ceil;
    for(let i=0;i<n;i++){
      let l=iL?iL[i]:0,r=iR?iR[i]:0;
      if(compOn){const x=Math.max(Math.abs(l),Math.abs(r));
        this.env=x>this.env?this.att*this.env+(1-this.att)*x:this.rel*this.env+(1-this.rel)*x;
        const over=20*Math.log10(this.env+1e-9)-thr;let gr=0;
        if(2*over<=-W)gr=0;else if(2*Math.abs(over)<W)gr=slope*(over+W/2)*(over+W/2)/(2*W);else gr=slope*over;
        if(gr<0){const g=Math.pow(10,gr/20);l*=g;r*=g;if(g<this.minC)this.minC=g;}}
      if(sat>0.001){l=(1-sat)*l+sat*Math.tanh(k*l)*tk;r=(1-sat)*r+sat*Math.tanh(k*r)*tk;}
      l*=this.drv;r*=this.drv;
      const pk=Math.max(Math.abs(l),Math.abs(r));const req=pk>c?c/pk:1;const cur=this.n;
      while(this.qn>0){const b=(this.qh+this.qn-1)%cap;if(this.qv[b]>=req)this.qn--;else break;}
      const t=(this.qh+this.qn)%cap;this.qv[t]=req;this.qp[t]=cur;this.qn++;
      while(this.qp[this.qh]<cur-la){this.qh=(this.qh+1)%cap;this.qn--;}
      const held=this.qv[this.qh];
      if(held<this.g)this.g=held;else this.g=held+(this.g-held)*this.lrel;
      this.gsum+=this.g-this.gb[this.gi];this.gb[this.gi]=this.g;this.gi=(this.gi+1)%la;
      let sm=this.gsum/la;if(sm>1)sm=1;
      const dl=this.dL[this.di],dr=this.dR[this.di];this.dL[this.di]=l;this.dR[this.di]=r;this.di=(this.di+1)%la;
      let yl=dl*sm,yr=dr*sm;if(yl>c)yl=c;else if(yl<-c)yl=-c;if(yr>c)yr=c;else if(yr<-c)yr=-c;
      oL[i]=yl;if(oR)oR[i]=yr;if(sm<this.minL)this.minL=sm;
      this.n++;if((this.n&65535)===0){let s=0;for(let j=0;j<la;j++)s+=this.gb[j];this.gsum=s;}
    }
    if(this.report){this.cnt+=n;if(this.cnt>=this.repN){this.port.postMessage({c:this.minC,l:this.minL});this.minC=1;this.minL=1;this.cnt=0;}}
    return true;}
}
registerProcessor('master-dyn',MasterDyn);
class KMeter extends AudioWorkletProcessor{
  constructor(){super();const[s1,s2]=kCoefs(sampleRate);this.s1=s1;this.s2=s2;this.z=new Float64Array(8);
    this.seg=Math.round(sampleRate*0.1);this.acc=0;this.cnt=0;this.ring=new Float64Array(30);this.ri=0;this.filled=0;this.pk=0;}
  process(inputs,outputs){const inp=inputs[0],out=outputs[0];const n=out[0].length;const s1=this.s1,s2=this.s2,z=this.z;
    for(let ch=0;ch<2;ch++){const x=inp&&(inp[ch]||inp[0]);const o=out[ch];if(!o)continue;const b=ch*4;
      for(let i=0;i<n;i++){const v=x?x[i]:0;o[i]=v;const a=Math.abs(v);if(a>this.pk)this.pk=a;
        const y=s1[0]*v+z[b];z[b]=s1[1]*v-s1[3]*y+z[b+1];z[b+1]=s1[2]*v-s1[4]*y;
        const w=s2[0]*y+z[b+2];z[b+2]=s2[1]*y-s2[3]*w+z[b+3];z[b+3]=s2[2]*y-s2[4]*w;
        this.acc+=w*w;}}
    this.cnt+=n;
    if(this.cnt>=this.seg){this.ring[this.ri]=this.acc/this.cnt;this.ri=(this.ri+1)%30;this.filled=Math.min(30,this.filled+1);this.acc=0;this.cnt=0;
      let m=0,s=0;for(let j=0;j<this.filled;j++){const e=this.ring[(this.ri-1-j+30)%30];if(j<4)m+=e;s+=e;}
      m/=Math.min(4,this.filled);s/=this.filled;
      this.port.postMessage({M:-0.691+10*Math.log10(m+1e-20),S:-0.691+10*Math.log10(s+1e-20),pk:this.pk});this.pk=0;}
    return true;}
}
registerProcessor('k-meter',KMeter);`;
const WURL = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
// Pages opened straight from disk (file://) cannot load worklets from blob: URLs in Chromium; a data: URL works there.
const WURL_DATA = 'data:application/javascript;base64,' + btoa(unescape(encodeURIComponent(WORKLET)));
async function addWorklet(ctx) {
  try { await addWorklet(ctx); } catch (e) { await ctx.audioWorklet.addModule(WURL_DATA); }
}

/* ---------- Controls and presets ---------- */
const hz = v => v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 1 : 2) + ' kHz' : Math.round(v) + ' Hz';
const dB = v => sgn(v) + ' dB';
const ms = v => Math.round(v) + ' ms';
const pct = v => Math.round(v * 100) + '%';
const SECTIONS = [
  { id: 'cleanup', title: 'Cleanup', note: 'Generated vocals and cymbals often carry a metallic edge between 3 and 6 kHz, and a brittle haze above 16 kHz. These bands trim both, along with rumble and boxiness.' },
  { id: 'tone', title: 'Tone', note: 'Broad moves for overall balance. A dB or two usually goes a long way.' },
  { id: 'stereo', title: 'Stereo', note: 'Widens the sides while keeping low end in mono, which holds up on phones and club systems.' },
  { id: 'dynamics', title: 'Glue and color', note: 'A gentle bus compressor and soft saturation ahead of the limiter.' },
  { id: 'loudness', title: 'Loudness', note: 'Pick where the master should land. Fit to target sets the limiter drive by rendering and measuring the whole track.' }
];
const CONTROLS = [
  { sec: 'cleanup', id: 'trim', label: 'Input trim', min: -12, max: 12, step: 0.1, fmt: dB, zero: 0, tip: 'Turns the song up or down before any processing. Use it if the glue compressor is working too hard, or not at all.' },
  { sec: 'cleanup', id: 'hpf', label: 'Low cut', min: 10, max: 120, step: 1, fmt: v => v <= 10 ? 'Off' : hz(v), tip: 'Removes very low rumble that you feel more than hear. 20 to 40 Hz is safe for most songs.' },
  { sec: 'cleanup', id: 'mudGain', label: 'Mud', min: -6, max: 3, step: 0.1, fmt: dB, zero: 0, tip: 'Cuts or boosts the low-mid band. Cutting a little clears up a boxy, muddy sound.' },
  { sec: 'cleanup', id: 'mudFreq', label: 'Mud frequency', min: 150, max: 600, step: 5, fmt: hz, tip: 'Where the mud band sits. Sweep it while listening for the spot where the song sounds clearest.' },
  { sec: 'cleanup', id: 'harshGain', label: 'Harshness', min: -8, max: 2, step: 0.1, fmt: dB, zero: 0, tip: 'Turns down the harsh band. If vocals start to sound dull or lispy, move it back toward 0.' },
  { sec: 'cleanup', id: 'harshFreq', label: 'Harshness frequency', min: 2000, max: 8000, step: 50, fmt: hz, tip: 'Where the harsh band sits. Sweep slowly; at the right spot the metallic ring drops away.' },
  { sec: 'cleanup', id: 'fizz', label: 'Top-end fizz cut', min: 12000, max: 22000, step: 100, fmt: v => v >= 22000 ? 'Off' : hz(v), tip: 'Gently rolls off the very top of the sound. Lower settings sound darker and smoother.' },
  { sec: 'tone', id: 'low', label: 'Low shelf, 100 Hz', min: -6, max: 6, step: 0.1, fmt: dB, zero: 0, tip: 'Adds or removes weight below about 100 Hz: kick drum, bass, body.' },
  { sec: 'tone', id: 'pres', label: 'Presence, 2.8 kHz', min: -4, max: 4, step: 0.1, fmt: dB, zero: 0, tip: 'Pushes vocals and lead instruments forward, or pulls them back.' },
  { sec: 'tone', id: 'air', label: 'Air shelf, 12 kHz', min: -4, max: 6, step: 0.1, fmt: dB, zero: 0, tip: 'Adds or removes shimmer and openness above about 12 kHz.' },
  { sec: 'stereo', id: 'width', label: 'Width', min: 0, max: 1.8, step: 0.01, fmt: pct, zero: 1, tip: '100% leaves the stereo image as it was made. Much above 130% can start to sound hollow.' },
  { sec: 'stereo', id: 'monoBass', label: 'Mono bass', min: 0, max: 250, step: 5, fmt: v => v <= 0 ? 'Off' : 'below ' + hz(v), tip: 'Below this frequency the left and right channels are made identical, so the low end stays centered and solid.' },
  { sec: 'dynamics', id: 'compThr', label: 'Glue threshold', min: -36, max: 0, step: 0.5, fmt: v => num(v) + ' dB', tip: 'The level where compression begins. Lower settings compress more of the song.' },
  { sec: 'dynamics', id: 'compRatio', label: 'Glue ratio', min: 1, max: 6, step: 0.1, fmt: v => v <= 1.001 ? 'Off' : v.toFixed(1) + ':1', tip: 'How firmly levels above the threshold are turned down. 2:1 is gentle; 4:1 and up is strong.' },
  { sec: 'dynamics', id: 'compAtt', label: 'Attack', min: 1, max: 100, step: 1, fmt: ms, tip: 'How fast the compressor reacts. Slower settings let drum hits punch through before it acts.' },
  { sec: 'dynamics', id: 'compRel', label: 'Release', min: 30, max: 800, step: 10, fmt: ms, tip: 'How fast it lets go. Too fast can sound pumping; too slow can sound flat and lifeless.' },
  { sec: 'dynamics', id: 'sat', label: 'Saturation', min: 0, max: 1, step: 0.01, fmt: pct, tip: 'Adds harmonic warmth and density, as analog tape and consoles do. Subtle settings usually work best.' },
  { sec: 'loudness', id: 'drive', label: 'Limiter drive', min: -12, max: 24, step: 0.1, fmt: dB, zero: 0, tip: 'How hard the song is pushed into the limiter. More drive means louder. Fit to target sets this for you.' },
  { sec: 'loudness', id: 'ceiling', label: 'Ceiling', min: -3, max: -0.1, step: 0.1, fmt: v => num(v) + ' dBFS', tip: 'The highest level the master may reach. −1 dBFS leaves room for MP3 and streaming conversion.' },
  { sec: 'loudness', id: 'limRel', label: 'Limiter release', min: 10, max: 400, step: 5, fmt: ms, tip: 'How quickly the limiter recovers after a peak. Shorter is louder but can add distortion.' }
];
const HELP = {
  preset: { paras: ['Starting points for common kinds of song. Choosing one sets every control below; adjust from there by ear.'] },
  deck: { title: 'Listen and compare', paras: [
    'This is your song. The grey shape is the original; the orange outline is the master after processing. Click anywhere on the waveform to jump there.',
    'Press Play, then switch between Original and Master to hear what the processing does. The B key does the same.',
    'Level-matched A/B plays the original at the same loudness as the master. Louder almost always sounds better at first, so matching the level lets you judge the tone and not the volume.'] },
  cleanup: { title: 'Cleanup', paras: [
    'Fixes problems before anything else. Low cut removes inaudible rumble. Mud thins out a boxy, cardboard sound in the low mids. Harshness tames the sharp, metallic ring that AI-generated vocals and cymbals often have. Fizz cut softens hiss and sizzle at the very top.',
    'Start here. Cutting problems usually helps a song more than boosting the good parts.'] },
  tone: { title: 'Tone', paras: [
    'Shapes the overall color of the song, like the bass and treble knobs on a stereo but more precise. Low shelf adds or removes weight, Presence brings vocals forward, and Air adds sparkle.',
    'Small moves of 1 to 2 dB are normal. If you find yourself going past 4 dB, the problem is probably better fixed in Cleanup.'] },
  stereo: { title: 'Stereo', paras: [
    'Controls how wide the song feels between the left and right speakers. Width above 100% spreads it out; below 100% narrows it.',
    'Mono bass keeps the lowest notes centered, so bass and kick stay solid on phones, Bluetooth speakers and club systems.'] },
  dynamics: { title: 'Glue and color', paras: [
    'The compressor gently evens out loud and quiet moments so the parts of the mix feel like one performance. Threshold sets where it starts working and ratio sets how firmly. Attack and release set how quickly it reacts and lets go.',
    'Saturation adds subtle warmth and density. Watch the Glue GR meter while playing: 1 to 3 dB of reduction is typical for mastering.'] },
  loudness: { title: 'Loudness', paras: [
    'The final stage. A limiter raises the overall level while stopping any peak from passing the ceiling, which would cause distortion.',
    'Pick a target. Spotify, YouTube and most services play songs at about \u221214 LUFS and turn louder songs down, so pushing much louder mostly costs punch. Fit to target sets the drive to land on your choice.'] },
  spectrum: { title: 'Spectrum', paras: [
    'Shows how much energy the song has at each frequency while it plays, from deep bass on the left to treble on the right. Grey is the original; orange is what you hear.',
    'The display is tilted so a well-balanced mix looks roughly level. Use it to see what your EQ moves are doing.'] },
  meters: { title: 'Output meters', paras: [
    'Live readings while the song plays. Momentary and Short-term show loudness in LUFS over the last 0.4 and 3 seconds; the tick marks your target. Peak shows the highest level and turns red near 0.',
    'Glue GR and Limiter GR show how many dB the compressor and limiter are turning the sound down. Limiter reduction that stays above about 6 dB can make a song sound squashed.'] },
  report: { title: 'Loudness report', paras: [
    'Measurements of the whole song, before and after. Integrated is the overall loudness that streaming services use. True peak is the highest level, including peaks that form between samples; distributors usually ask for \u22121 dBTP or lower.',
    'Loudness range is how much the level varies between quiet and loud sections. Peak to loudness is a rough guide to punch: higher means more dynamic.',
    'Fit to target renders the song and sets the drive for you. Measure as set renders it with the current settings, unchanged.'] },
  export: { title: 'Export', paras: [
    'Saves the finished master. 24-bit WAV is best for distributors and for archiving. 16-bit with dither is CD format.',
    'Save settings stores every control position in a small file, so you can reuse them on other songs or in the Python batch script.'] }
};
const TARGETS = [[-16, 'Apple Music'], [-14, 'Spotify, YouTube'], [-11, 'Loud'], [-9, 'Club, very loud']];
const BASE = { trim: 0, hpf: 30, mudGain: -1.5, mudFreq: 280, harshGain: -2, harshFreq: 3800, fizz: 18000, low: 1, pres: 0, air: 1.5,
  width: 1.1, monoBass: 120, compThr: -20, compRatio: 2, compAtt: 30, compRel: 200, sat: 0.15, ceiling: -1, limRel: 80, target: -14 };
const PRESETS = {
  cleanup: { name: 'Suno cleanup', v: { ...BASE } },
  loud: { name: 'Loud and bright', v: { ...BASE, harshGain: -2.5, air: 2, width: 1.15, compThr: -22, compRatio: 2.5, sat: 0.3, limRel: 60, target: -9 } },
  acoustic: { name: 'Acoustic and vocal', v: { ...BASE, hpf: 40, mudGain: -1, harshGain: -3, harshFreq: 4200, fizz: 19000, low: 0, pres: 0.5, air: 1, width: 1, monoBass: 100, compThr: -24, compRatio: 1.5, compAtt: 40, compRel: 300, sat: 0.05, limRel: 120, target: -16 } },
  warm: { name: 'Warm and dark', v: { ...BASE, harshGain: -3.5, harshFreq: 3400, fizz: 15500, low: 2, air: 0, sat: 0.3 } },
  flat: { name: 'Limiter only', v: { ...BASE, hpf: 10, mudGain: 0, harshGain: 0, fizz: 22000, low: 0, pres: 0, air: 0, width: 1, monoBass: 0, compRatio: 1, sat: 0 } }
};
let presetId = 'cleanup';
const S = { ...PRESETS.cleanup.v, drive: 0 };
const keyOf = () => JSON.stringify(S);

/* ---------- DSP graph (shared by live and offline) ---------- */
function dynParams(s) {
  return { compThr: s.compThr, compRatio: s.compRatio, compAtt: s.compAtt / 1000, compRel: s.compRel / 1000, knee: 6,
    sat: s.sat, drive: s.drive, ceiling: s.ceiling, limRel: s.limRel / 1000 };
}
function buildChain(ctx, s, live) {
  const bq = (type) => { const f = ctx.createBiquadFilter(); f.type = type; return f; };
  const g = (v = 1) => { const n = ctx.createGain(); n.gain.value = v; return n; };
  const N = { input: g(), hpf: bq('highpass'), mud: bq('peaking'), harsh: bq('peaking'), low: bq('lowshelf'), pres: bq('peaking'), air: bq('highshelf'), fizz: bq('lowpass') };
  N.input.connect(N.hpf).connect(N.mud).connect(N.harsh).connect(N.low).connect(N.pres).connect(N.air).connect(N.fizz);
  const split = ctx.createChannelSplitter(2), merge = ctx.createChannelMerger(2);
  N.fizz.connect(split);
  const mL = g(0.5), mR = g(0.5), sL = g(0.5), sR = g(-0.5), mid = g(), side = g();
  split.connect(mL, 0); split.connect(mR, 1); split.connect(sL, 0); split.connect(sR, 1);
  mL.connect(mid); mR.connect(mid); sL.connect(side); sR.connect(side);
  N.sh1 = bq('highpass'); N.sh2 = bq('highpass'); N.sw = g(1); const neg = g(-1);
  side.connect(N.sh1).connect(N.sh2).connect(N.sw);
  mid.connect(merge, 0, 0); mid.connect(merge, 0, 1); N.sw.connect(merge, 0, 0); N.sw.connect(neg).connect(merge, 0, 1);
  N.dyn = new AudioWorkletNode(ctx, 'master-dyn', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], channelCount: 2,
    channelCountMode: 'explicit', processorOptions: { params: dynParams(s), report: live } });
  merge.connect(N.dyn);
  N.output = N.dyn;
  applyParams(N, s, ctx, false);
  return N;
}
function applyParams(N, s, ctx, smooth) {
  const set = (p, v) => { if (smooth) p.setTargetAtTime(v, ctx.currentTime, 0.02); else p.value = v; };
  set(N.input.gain, db2lin(s.trim));
  set(N.hpf.frequency, s.hpf); N.hpf.Q.value = BUTTER_DB;
  set(N.mud.frequency, s.mudFreq); N.mud.Q.value = 1.0; set(N.mud.gain, s.mudGain);
  set(N.harsh.frequency, s.harshFreq); N.harsh.Q.value = 1.6; set(N.harsh.gain, s.harshGain);
  N.low.frequency.value = 100; set(N.low.gain, s.low);
  N.pres.frequency.value = 2800; N.pres.Q.value = 0.7; set(N.pres.gain, s.pres);
  N.air.frequency.value = 12000; set(N.air.gain, s.air);
  set(N.fizz.frequency, s.fizz >= 22000 ? ctx.sampleRate / 2 : s.fizz); N.fizz.Q.value = BUTTER_DB;
  const mb = s.monoBass <= 0 ? 10 : s.monoBass;
  set(N.sh1.frequency, mb); set(N.sh2.frequency, mb); N.sh1.Q.value = N.sh2.Q.value = BUTTER_DB;
  set(N.sw.gain, s.width);
  if (smooth) N.dyn.port.postMessage(dynParams(s));
}
async function renderMaster(buf, s) {
  const sr = buf.sampleRate, la = Math.max(1, Math.round(sr * 0.005));
  const ctx = new OfflineAudioContext(2, buf.length + la, sr);
  await addWorklet(ctx);
  const N = buildChain(ctx, s, false);
  const src = ctx.createBufferSource(); src.buffer = buf; src.connect(N.input); N.output.connect(ctx.destination); src.start();
  const out = await ctx.startRendering();
  const res = new AudioBuffer({ numberOfChannels: 2, length: buf.length, sampleRate: sr });
  for (let c = 0; c < 2; c++) res.copyToChannel(out.getChannelData(c).subarray(la, la + buf.length), c);
  return res;
}

/* ---------- Analysis ---------- */
const TP_TAB = (() => { const t = [];
  for (let p = 1; p < 4; p++) { const fr = p / 4, h = new Float64Array(16); let s = 0;
    for (let j = 0; j < 16; j++) { const tt = fr - (j - 7); const sinc = Math.sin(Math.PI * tt) / (Math.PI * tt); const u = (tt + 8) / 16;
      h[j] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u)); s += h[j]; }
    for (let j = 0; j < 16; j++) h[j] /= s; t.push(h); }
  return t; })();
function truePeak(buf) {
  let m = 0;
  for (let c = 0; c < buf.numberOfChannels; c++) { const x = buf.getChannelData(c); for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; } }
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const x = buf.getChannelData(c), n = x.length;
    for (let i = 7; i < n - 9; i++) {
      if (Math.max(Math.abs(x[i]), Math.abs(x[i + 1])) < m * 0.6) continue;
      for (let p = 0; p < 3; p++) { const h = TP_TAB[p]; let y = 0; for (let j = 0; j < 16; j++) y += h[j] * x[i + j - 7]; y = Math.abs(y); if (y > m) m = y; }
    }
  }
  return lin2db(m);
}
function analyze(buf, withTP) {
  const sr = buf.sampleRate, seg = Math.round(sr * 0.1), nSeg = Math.floor(buf.length / seg);
  const E = new Float64Array(nSeg); const [s1, s2] = kCoefs(sr); let sp = 0;
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const x = buf.getChannelData(c); let a1 = 0, a2 = 0, b1 = 0, b2 = 0, acc = 0, cnt = 0, si = 0;
    const lim = nSeg * seg;
    for (let i = 0; i < x.length; i++) {
      const v = x[i], av = v < 0 ? -v : v; if (av > sp) sp = av;
      if (i >= lim) continue;
      const y = s1[0] * v + a1; a1 = s1[1] * v - s1[3] * y + a2; a2 = s1[2] * v - s1[4] * y;
      const w = y + b1; b1 = -2 * y - s2[3] * w + b2; b2 = y - s2[4] * w;
      acc += w * w; if (++cnt === seg) { E[si++] += acc / seg; acc = 0; cnt = 0; }
    }
  }
  const L = e => -0.691 + 10 * Math.log10(e + 1e-20);
  const P = new Float64Array(nSeg + 1); for (let i = 0; i < nSeg; i++) P[i + 1] = P[i] + E[i];
  const win = (b, k) => (P[b + k] - P[b]) / k;
  let blk = []; for (let b = 0; b + 4 <= nSeg; b++) { const e = win(b, 4); if (L(e) > -70) blk.push(e); }
  let I = -Infinity;
  if (blk.length) { const rel = L(blk.reduce((a, b) => a + b, 0) / blk.length) - 10; const g = blk.filter(e => L(e) > rel); if (g.length) I = L(g.reduce((a, b) => a + b, 0) / g.length); }
  let st = [], Smax = -Infinity;
  for (let b = 0; b + 30 <= nSeg; b++) { const e = win(b, 30); const l = L(e); if (l > Smax) Smax = l; if (l > -70) st.push(e); }
  let LRA = NaN;
  if (st.length > 2) { const rel = L(st.reduce((a, b) => a + b, 0) / st.length) - 20; const v = st.map(L).filter(l => l > rel).sort((a, b) => a - b);
    if (v.length > 1) LRA = v[Math.round(0.95 * (v.length - 1))] - v[Math.round(0.1 * (v.length - 1))]; }
  return { I, LRA, Smax, SP: lin2db(sp), TP: withTP ? truePeak(buf) : NaN };
}

/* ---------- File decoding ---------- */
function detectRate(ab) {
  const u = new Uint8Array(ab), dv = new DataView(ab), str = (o, n) => String.fromCharCode(...u.subarray(o, o + n));
  try {
    if (str(0, 4) === 'RIFF' && str(8, 4) === 'WAVE') { let o = 12; while (o + 8 < u.length) { const id = str(o, 4), sz = dv.getUint32(o + 4, true); if (id === 'fmt ') return dv.getUint32(o + 12, true); o += 8 + sz + (sz & 1); } }
    if (str(0, 4) === 'fLaC') return (u[18] << 12) | (u[19] << 4) | (u[20] >> 4);
    let o = 0; if (str(0, 3) === 'ID3') o = 10 + ((u[6] & 127) << 21 | (u[7] & 127) << 14 | (u[8] & 127) << 7 | (u[9] & 127));
    for (let i = o; i < Math.min(u.length - 4, o + 65536); i++) {
      if (u[i] === 0xFF && (u[i + 1] & 0xE0) === 0xE0) { const ver = (u[i + 1] >> 3) & 3, ri = (u[i + 2] >> 2) & 3, layer = (u[i + 1] >> 1) & 3;
        if (ri === 3 || ver === 1 || layer === 0) continue;
        const T = { 3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000] }; return T[ver][ri]; }
    }
  } catch (e) {}
  return null;
}
function toStereo(buf) {
  if (buf.numberOfChannels === 2) return buf;
  const out = new AudioBuffer({ numberOfChannels: 2, length: buf.length, sampleRate: buf.sampleRate });
  out.copyToChannel(buf.getChannelData(0), 0); out.copyToChannel(buf.getChannelData(buf.numberOfChannels > 1 ? 1 : 0), 1);
  return out;
}

/* ---------- WAV encoding ---------- */
function encodeWav(buf, bits) {
  const ch = buf.numberOfChannels, n = buf.length, sr = buf.sampleRate, bps = bits / 8, data = n * ch * bps;
  const u = new Uint8Array(44 + data), dv = new DataView(u.buffer);
  const w4 = (o, s) => { for (let i = 0; i < 4; i++) u[o + i] = s.charCodeAt(i); };
  w4(0, 'RIFF'); dv.setUint32(4, 36 + data, true); w4(8, 'WAVE'); w4(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, ch, true); dv.setUint32(24, sr, true); dv.setUint32(28, sr * ch * bps, true); dv.setUint16(32, ch * bps, true); dv.setUint16(34, bits, true);
  w4(36, 'data'); dv.setUint32(40, data, true);
  const chs = []; for (let c = 0; c < ch; c++) chs.push(buf.getChannelData(c));
  let o = 44;
  if (bits === 24) {
    for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { let v = Math.round(clamp(chs[c][i], -1, 1) * 8388607); if (v < 0) v += 16777216; u[o++] = v & 255; u[o++] = (v >> 8) & 255; u[o++] = (v >> 16) & 255; }
  } else {
    for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { const v = clamp(Math.round(chs[c][i] * 32767 + (Math.random() - Math.random())), -32768, 32767); dv.setInt16(o, v, true); o += 2; }
  }
  return u;
}

/* ---------- Demo loop ---------- */
async function makeDemo() {
  const sr = 44100, bpm = 96, beat = 60 / bpm, bars = 8, dur = bars * 4 * beat;
  const ctx = new OfflineAudioContext(2, Math.round(dur * sr), sr);
  const bus = ctx.createGain(); bus.gain.value = 0.5; bus.connect(ctx.destination);
  const nb = ctx.createBuffer(1, sr, sr); const nd = nb.getChannelData(0); for (let i = 0; i < sr; i++) nd[i] = Math.random() * 2 - 1;
  const mf = m => 440 * Math.pow(2, (m - 69) / 12);
  const chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]], roots = [33, 29, 36, 31];
  const env = (t, a, d, lv) => { const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(lv, t + a); g.gain.exponentialRampToValueAtTime(0.0001, t + a + d); return g; };
  const noise = (t, dur, type, f, lv, pan) => { const s = ctx.createBufferSource(); s.buffer = nb; const fl = ctx.createBiquadFilter(); fl.type = type; fl.frequency.value = f;
    const g = env(t, 0.002, dur, lv); const p = ctx.createStereoPanner(); p.pan.value = pan; s.connect(fl).connect(g).connect(p).connect(bus); s.start(t, Math.random() * 0.5); s.stop(t + dur + 0.05); };
  for (let bar = 0; bar < bars; bar++) {
    const t0 = bar * 4 * beat, ci = Math.floor(bar / 2) % 4;
    for (const b of [0, 2, bar % 2 ? 2.5 : -1]) { if (b < 0) continue; const t = t0 + b * beat; const o = ctx.createOscillator(); o.frequency.setValueAtTime(140, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.15);
      const g = env(t, 0.003, 0.35, 0.9); o.connect(g).connect(bus); o.start(t); o.stop(t + 0.4); }
    for (const b of [1, 3]) { const t = t0 + b * beat; noise(t, 0.18, 'highpass', 1200, 0.35, 0); const o = ctx.createOscillator(); o.type = 'triangle'; o.frequency.value = 190;
      const g = env(t, 0.002, 0.1, 0.2); o.connect(g).connect(bus); o.start(t); o.stop(t + 0.15); }
    for (let e = 0; e < 8; e++) noise(t0 + e * beat / 2, 0.045, 'highpass', 8000, e % 2 ? 0.12 : 0.07, 0.3);
    for (let b = 0; b < 4; b++) { const t = t0 + b * beat; const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = mf(roots[ci] + 12 * (b === 3 ? 1 : 0));
      const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 380; const g = env(t, 0.01, beat * 0.85, 0.22); o.connect(f).connect(g).connect(bus); o.start(t); o.stop(t + beat); }
    if (bar % 2 === 0) { const t = t0, len = 8 * beat;
      for (const m of chords[ci]) for (const [det, pan] of [[-7, -0.7], [7, 0.7]]) {
        const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = mf(m); o.detune.value = det;
        const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 1700; f.Q.value = 0.5;
        const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.035, t + 0.4); g.gain.setValueAtTime(0.035, t + len - 0.5); g.gain.linearRampToValueAtTime(0, t + len - 0.02);
        const p = ctx.createStereoPanner(); p.pan.value = pan; o.connect(f).connect(g).connect(p).connect(bus); o.start(t); o.stop(t + len); }
    }
  }
  const buf = await ctx.startRendering();
  const a = analyze(buf, false); const gain = Math.min(db2lin(-19 - a.I), 0.9 / db2lin(a.SP));
  for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] *= gain; }
  return buf;
}

/* ---------- Help tooltips ---------- */
const tipEl = $('#tip'); let tipTimer = 0, tipOwner = null;
function placeTip(anchor) {
  const r = anchor.getBoundingClientRect(), tw = tipEl.offsetWidth, th = tipEl.offsetHeight, vw = innerWidth, vh = innerHeight;
  const left = clamp(r.left, 12, Math.max(12, vw - tw - 12));
  let top = r.bottom + 8; if (top + th > vh - 12 && r.top - th - 8 > 12) top = r.top - th - 8;
  tipEl.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}
function attachTip(anchor, content, btn) {
  const show = () => {
    clearTimeout(tipTimer); if (tipOwner && tipOwner !== anchor) tipOwner._hideTip();
    tipEl.replaceChildren();
    if (content.title) { const t = document.createElement('strong'); t.textContent = content.title; tipEl.appendChild(t); }
    content.paras.forEach(x => { const p = document.createElement('p'); p.textContent = x; tipEl.appendChild(p); });
    tipEl.hidden = false; placeTip(anchor); tipOwner = anchor; if (btn) btn.setAttribute('aria-expanded', 'true');
  };
  const hide = () => { clearTimeout(tipTimer); if (tipOwner !== anchor) return; tipEl.hidden = true; tipOwner = null; if (btn) btn.setAttribute('aria-expanded', 'false'); };
  anchor._hideTip = hide;
  anchor.addEventListener('pointerenter', e => { if (e.pointerType !== 'mouse') return; clearTimeout(tipTimer); tipTimer = setTimeout(show, 250); });
  anchor.addEventListener('pointerleave', e => { if (e.pointerType !== 'mouse') return; clearTimeout(tipTimer); if (!(btn && document.activeElement === btn)) hide(); });
  if (btn) { btn.addEventListener('focus', show); btn.addEventListener('blur', hide); btn.addEventListener('click', e => { e.stopPropagation(); show(); }); }
}
document.addEventListener('keydown', e => { if (e.key === 'Escape' && tipOwner) tipOwner._hideTip(); });
document.addEventListener('pointerdown', e => { if (tipOwner && !tipOwner.contains(e.target)) tipOwner._hideTip(); });
addEventListener('scroll', () => { if (tipOwner) placeTip(tipOwner); }, { passive: true });
addEventListener('resize', () => { if (tipOwner) tipOwner._hideTip(); });
function addPanelHelp() {
  document.querySelectorAll('[data-help]').forEach(sec => {
    const h = sec.querySelector('.modhead h2'), help = HELP[sec.dataset.help]; if (!h || !help) return;
    const wrap = document.createElement('span'); wrap.className = 'ht'; h.replaceWith(wrap); wrap.appendChild(h);
    const b = document.createElement('button'); b.type = 'button'; b.className = 'help'; b.textContent = '?';
    b.setAttribute('aria-label', 'What ' + help.title + ' does'); b.setAttribute('aria-expanded', 'false');
    wrap.appendChild(b); attachTip(wrap, help, b);
  });
  const pl = document.querySelector('label[data-tip="preset"]'); attachTip(pl, HELP.preset);
}

/* ---------- UI build ---------- */
const rack = $('#rack');
const inputs = {};
SECTIONS.forEach((sec, si) => {
  const m = document.createElement('section'); m.className = 'panel mod'; m.dataset.help = sec.id;
  m.innerHTML = `<div class="modhead"><h2>${sec.title}</h2><span class="idx">${String(si + 1).padStart(2, '0')}</span></div><p class="note">${sec.note}</p>`;
  if (sec.id === 'loudness') {
    const tg = document.createElement('div'); tg.className = 'targets'; tg.setAttribute('role', 'group'); tg.setAttribute('aria-label', 'Loudness target');
    TARGETS.forEach(([v, l]) => { const b = document.createElement('button'); b.type = 'button'; b.dataset.t = v; b.innerHTML = `<b>${num(v, 0)} LUFS</b><span>${l}</span>`;
      b.addEventListener('click', () => setTarget(v)); tg.appendChild(b); });
    m.appendChild(tg);
  }
  CONTROLS.filter(c => c.sec === sec.id).forEach(c => {
    const d = document.createElement('div'); d.className = 'ctl';
    d.innerHTML = `<label for="c-${c.id}"${c.tip ? ' data-tip' : ''}>${c.label}</label><output id="o-${c.id}" for="c-${c.id}"></output><input type="range" id="c-${c.id}" min="${c.min}" max="${c.max}" step="${c.step}">`;
    m.appendChild(d);
    const el = d.querySelector('input'); inputs[c.id] = el;
    if (c.tip) attachTip(d.querySelector('label'), { title: c.label, paras: [c.tip] });
    el.addEventListener('input', () => { S[c.id] = +el.value; paint(c); onChange(); });
    el.addEventListener('dblclick', () => { const v = c.id === 'drive' ? 0 : PRESETS[presetId].v[c.id]; S[c.id] = v; syncControl(c); onChange(); });
  });
  rack.appendChild(m);
});
function paint(c) {
  const el = inputs[c.id], v = +el.value, p = (x) => ((x - c.min) / (c.max - c.min) * 100);
  const z = c.zero !== undefined ? p(c.zero) : 0, pv = p(v);
  el.style.setProperty('--a', Math.min(z, pv) + '%'); el.style.setProperty('--b', Math.max(z, pv) + '%');
  $('#o-' + c.id).textContent = c.fmt(v);
}
function syncControl(c) { inputs[c.id].value = S[c.id]; paint(c); }
function syncAll() {
  CONTROLS.forEach(syncControl);
  document.querySelectorAll('.targets button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.t === S.target ? 'true' : 'false'));
  const tp = clamp((S.target + 40) / 40 * 100, 0, 100) + '%'; $('#tg-m').style.left = tp; $('#tg-s').style.left = tp;
}
addPanelHelp();
const psel = $('#preset');
Object.entries(PRESETS).forEach(([k, p]) => { const o = document.createElement('option'); o.value = k; o.textContent = p.name; psel.appendChild(o); });
psel.addEventListener('change', () => {
  presetId = psel.value; const old = S.target; Object.assign(S, PRESETS[presetId].v); S.drive = clamp(S.drive + (S.target - old), -12, 24);
  syncAll(); onChange();
});
function setTarget(v) { S.drive = clamp(S.drive + (v - S.target), -12, 24); S.target = v; syncAll(); onChange(); }

/* ---------- State and status ---------- */
let track = null, master = null, busy = false;
const statusEl = $('#status');
function status(t, isBusy = false) { statusEl.textContent = t; statusEl.classList.toggle('busy', isBusy); }
function fresh() { return master && master.key === keyOf(); }
function onChange() { if (L) { applyParams(L.chain, S, L.ctx, true); updateAB(); } renderReport(); }
function setBusy(b) { busy = b; rack.disabled = b; ['#fitBtn', '#measureBtn', '#saveBtn', '#demoBtn', '#preset'].forEach(s => $(s).disabled = b); $('#file').disabled = b; }

function renderReport() {
  const o = track && track.an, m = master && master.an, f = fresh();
  const rows = [['Integrated', 'I', ' LUFS'], ['Short-term max', 'Smax', ' LUFS'], ['True peak', 'TP', ' dBTP'], ['Sample peak', 'SP', ' dBFS'], ['Loudness range', 'LRA', ' LU'], ['Peak to loudness', 'PLR', ' dB']];
  const val = (a, k) => { if (!a) return '–'; const v = k === 'PLR' ? a.TP - a.I : a[k]; return isNaN(v) ? '–' : num(v); };
  $('#repBody').innerHTML = rows.map(([l, k, u]) => `<tr><td>${l}</td><td>${val(o, k)}</td><td>${val(m, k)}</td></tr>`).join('');
  $('#report').classList.toggle('stale', !!m && !f);
  $('#staleFlag').hidden = !m || f;
  const notes = [];
  if (o) {
    if (o.TP > -0.2) notes.push(`The original reaches ${num(o.TP)} dBTP, which suggests it was already limited or clipped on export. Extra limiting will add little loudness.`);
    if (o.I > -11) notes.push(`The original already sits at ${num(o.I)} LUFS. Streaming services turn louder tracks down to about ${MINUS}14 LUFS, so a more open master may sound better after normalization.`);
  }
  if (m && f) {
    if (m.TP > S.ceiling + 0.3) notes.push(`Inter-sample peaks reach ${num(m.TP)} dBTP. Lower the ceiling a few tenths if a distributor asks for ${MINUS}1 dBTP.`);
    if (m.LRA < 3) notes.push(`Loudness range is narrow at ${num(m.LRA)} LU. Ease the glue compressor or the target if the track feels flat.`);
    if (Math.abs(m.I - S.target) > 0.5) notes.push(`The master measures ${num(m.I)} LUFS against a ${num(S.target, 0)} LUFS target. Fit to target closes the gap.`);
  }
  $('#notes').innerHTML = notes.map(n => `<li>${n}</li>`).join('');
}

/* ---------- Live playback ---------- */
let L = null, src = null, playing = false, offset = 0, startedAt = 0, abMode = 'master';
const meterVals = { M: -Infinity, S: -Infinity, pk: 0, c: 1, l: 1 };
async function ensureCtx() {
  if (L) return L;
  const ctx = new AudioContext({ latencyHint: 'playback' });
  await addWorklet(ctx);
  const g = () => ctx.createGain();
  const srcBus = g(), origG = g(), mastG = g(), mix = g();
  const chain = buildChain(ctx, S, true);
  const meter = new AudioWorkletNode(ctx, 'k-meter', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit' });
  const pre = ctx.createAnalyser(), post = ctx.createAnalyser();
  pre.fftSize = post.fftSize = 8192; pre.smoothingTimeConstant = post.smoothingTimeConstant = 0.82;
  srcBus.connect(origG).connect(mix); srcBus.connect(chain.input); chain.output.connect(mastG).connect(mix);
  mix.connect(meter).connect(ctx.destination); mix.connect(post); srcBus.connect(pre);
  chain.dyn.port.onmessage = e => { meterVals.c = Math.min(meterVals.c, e.data.c); meterVals.l = Math.min(meterVals.l, e.data.l); };
  meter.port.onmessage = e => { meterVals.M = e.data.M; meterVals.S = e.data.S; meterVals.pk = Math.max(meterVals.pk, e.data.pk); };
  L = { ctx, srcBus, origG, mastG, chain, pre, post, preD: new Float32Array(pre.frequencyBinCount), postD: new Float32Array(post.frequencyBinCount) };
  updateAB();
  return L;
}
function updateAB() {
  if (!L) return;
  const ref = fresh() ? master.an.I : S.target;
  const og = $('#match').checked && track && isFinite(track.an.I) ? db2lin(clamp(ref - track.an.I, -24, 24)) : 1;
  const t = L.ctx.currentTime;
  L.origG.gain.setTargetAtTime(abMode === 'orig' ? og : 0, t, 0.012);
  L.mastG.gain.setTargetAtTime(abMode === 'master' ? 1 : 0, t, 0.012);
}
function setAB(m) { abMode = m; $('#abOrig').setAttribute('aria-pressed', m === 'orig'); $('#abMaster').setAttribute('aria-pressed', m === 'master'); updateAB(); }
function pos() {
  if (!track) return 0; if (!playing || !L) return offset;
  const d = track.buffer.duration; let p = L.ctx.currentTime - startedAt;
  if ($('#loop').checked) p = ((p % d) + d) % d; return Math.min(p, d);
}
function startSrc(at) {
  if (src) { src.onended = null; try { src.stop(); } catch (e) {} }
  src = L.ctx.createBufferSource(); src.buffer = track.buffer; src.loop = $('#loop').checked; src.connect(L.srcBus);
  src.start(0, at); startedAt = L.ctx.currentTime - at;
  const me = src; src.onended = () => { if (src === me && playing) { playing = false; offset = 0; src = null; updatePlayBtn(); } };
}
async function play() {
  if (!track) return;
  await ensureCtx(); await L.ctx.resume();
  if (offset >= track.buffer.duration - 0.05) offset = 0;
  startSrc(offset); playing = true; updatePlayBtn();
}
function pause() { offset = pos(); playing = false; if (src) { src.onended = null; try { src.stop(); } catch (e) {} src = null; } updatePlayBtn(); }
function seek(t) { offset = clamp(t, 0, track.buffer.duration); if (playing) startSrc(offset); }
function updatePlayBtn() { $('#playBtn').textContent = playing ? 'Pause' : 'Play'; }

$('#playBtn').addEventListener('click', () => playing ? pause() : play());
$('#abOrig').addEventListener('click', () => setAB('orig'));
$('#abMaster').addEventListener('click', () => setAB('master'));
$('#match').addEventListener('change', updateAB);
$('#loop').addEventListener('change', () => { if (playing) startSrc(pos()); });
document.addEventListener('keydown', e => {
  const t = e.target; if (t.closest && t.closest('input,select,textarea,button')) return;
  if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
  if (e.key === 'b' || e.key === 'B') setAB(abMode === 'master' ? 'orig' : 'master');
});

/* ---------- Loading ---------- */
async function setTrack(buf, name, base) {
  if (playing) pause(); offset = 0;
  track = { buffer: toStereo(buf), name };
  master = null; peaksCache = {};
  $('#trackName').textContent = name;
  $('#trackMeta').textContent = `${fmtTime(track.buffer.duration)} · ${(track.buffer.sampleRate / 1000).toFixed(1)} kHz · stereo`;
  $('#fname').value = base + '_master';
  status('Measuring the original', true); await tick();
  track.an = analyze(track.buffer, true);
  S.drive = clamp(Math.round((S.target - track.an.I) * 10) / 10, -12, 24); syncAll();
  renderReport(); updateAB();
  await fit();
}
async function loadFile(file) {
  if (!file || busy) return;
  setBusy(true); status('Decoding ' + file.name, true);
  try {
    await decodeAndSet(await file.arrayBuffer(), file.name);
  } catch (e) {
    setBusy(false); status(`This file could not be decoded. Try a WAV or MP3 export from Suno.`);
  }
}
async function decodeAndSet(ab, name) {
  {
    const sr = detectRate(ab) || 44100;
    const dctx = new OfflineAudioContext(2, 1, sr);
    const buf = await dctx.decodeAudioData(ab);
    setBusy(false);
    await setTrack(buf, name, name.replace(/\.[^.]+$/, '').replace(/[^\w\- ]+/g, '').trim() || 'track');
  }
}
$('#file').addEventListener('change', e => { loadFile(e.target.files[0]); e.target.value = ''; });
$('#loadLbl').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); } });
$('#demoBtn').addEventListener('click', loadDemo);
async function loadDemo() {
  if (busy) return; setBusy(true); status('Building the demo loop', true);
  const b = await makeDemo(); setBusy(false);
  await setTrack(b, 'Demo loop, synthesized (8 bars)', 'demo-loop');
}
let dragDepth = 0;
document.addEventListener('dragenter', e => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
document.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; document.body.classList.remove('dragging'); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

/* ---------- Render, fit, export ---------- */
async function fit() {
  if (!track || busy) return; setBusy(true);
  try {
    const snap = { ...S }; let d = snap.drive, prev = null, buf = null, an = null;
    for (let i = 0; i < 6; i++) {
      status(`Fitting to ${num(snap.target, 0)} LUFS, pass ${i + 1}`, true); await tick();
      snap.drive = d; buf = await renderMaster(track.buffer, snap); an = analyze(buf, false);
      if (!isFinite(an.I)) break;
      const err = snap.target - an.I; if (Math.abs(err) < 0.1) break;
      let slope = 1; if (prev && Math.abs(d - prev.d) > 1e-3) { const s = (an.I - prev.I) / (d - prev.d); if (isFinite(s)) slope = clamp(s, 0.15, 1.2); }
      prev = { d, I: an.I }; const nd = clamp(d + err / slope, -12, 24); if (Math.abs(nd - d) < 0.01) break; d = nd;
    }
    status('Measuring true peak', true); await tick();
    an = analyze(buf, true);
    S.drive = Math.round(snap.drive * 100) / 100; snap.drive = S.drive;
    syncAll();
    master = { buffer: buf, an, key: keyOf() };
    peaksCache = {};
    onChange();
    status(`Master measures ${num(an.I)} LUFS integrated, ${num(an.TP)} dBTP true peak. Press Play and switch A/B to compare.`);
  } catch (e) { status('Rendering failed in this browser: ' + (e && e.message || e)); }
  setBusy(false);
}
async function measure() {
  if (!track || busy) return; setBusy(true);
  try {
    status('Rendering with current settings', true); await tick();
    const buf = await renderMaster(track.buffer, { ...S }); status('Measuring', true); await tick();
    const an = analyze(buf, true); master = { buffer: buf, an, key: keyOf() }; peaksCache = {}; onChange();
    status(`Master measures ${num(an.I)} LUFS integrated, ${num(an.TP)} dBTP true peak.`);
  } catch (e) { status('Rendering failed in this browser: ' + (e && e.message || e)); }
  setBusy(false);
}
$('#fitBtn').addEventListener('click', fit);
$('#measureBtn').addEventListener('click', measure);

const LOCAL = window.__SUNO_LOCAL__ || null;
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), 'X-Desk-Token': LOCAL.token } });
  if (!r.ok) throw { code: 'local_error', message: (await r.text()) || ('HTTP ' + r.status) };
  return r;
}
if (LOCAL) $('#saveNote').textContent = 'Writes the file straight to ' + LOCAL.outdir + '. Settings that changed since the last measurement are rendered first.';
else $('#saveNote').textContent = 'Downloads the file through your browser. Settings that changed since the last measurement are rendered first.';
async function saveFile(filename, data) {
  if (LOCAL) {
    const r = await api('/api/save?name=' + encodeURIComponent(filename), { method: 'POST', body: data });
    return 'Saved to ' + (await r.json()).path;
  }
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data]));
  const a = document.createElement('a'); a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return 'Downloaded ' + filename + '.';
}
function saveError(e) {
  const c = e && e.code;
  return c === 'local_error' ? 'Could not write the file: ' + e.message : 'Saving failed' + (e && e.message ? ': ' + e.message : '.');
}
const baseName = () => ($('#fname').value || 'master').replace(/[^\w\- .]+/g, '').trim() || 'master';
$('#saveSet').addEventListener('click', async () => {
  const name = baseName().replace(/_master$/, '') + '_settings.json';
  try { status(await saveFile(name, JSON.stringify({ preset: presetId, ...S }, null, 2))); } catch (e) { status(saveError(e)); }
});
$('#setLbl').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#setFile').click(); } });
$('#setFile').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const j = JSON.parse(await f.text()); let n = 0;
    if (j.preset && PRESETS[j.preset]) { presetId = j.preset; psel.value = j.preset; }
    for (const k of Object.keys(S)) if (typeof j[k] === 'number' && isFinite(j[k])) { S[k] = j[k]; n++; }
    syncAll(); onChange(); status(`Loaded ${n} settings from ${f.name}.` + (track ? ' Fit to target or Measure to update the report.' : ''));
  } catch (err) { status(f.name + ' is not a settings file saved from this desk.'); }
});
$('#saveBtn').addEventListener('click', async () => {
  if (!track || busy) return;
  if (!fresh()) await measure();
  if (!master) return;
  setBusy(true); status('Encoding WAV', true); await tick();
  const bits = +$('#bits').value, base = baseName();
  const wav = encodeWav(master.buffer, bits);
  setBusy(false);
  try { status(await saveFile(base + '.wav', new Blob([wav], { type: 'audio/wav' }))); } catch (e) { status(saveError(e)); }
});

/* ---------- Drawing ---------- */
const cs = {}; function readColors() { const st = getComputedStyle(document.documentElement); ['--screen', '--screen-grid', '--screen-ink', '--trace', '--trace-pre'].forEach(k => cs[k] = st.getPropertyValue(k).trim()); }
readColors();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); peaksCache = {}; });
new MutationObserver(() => { readColors(); peaksCache = {}; }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
function fitCanvas(cv) { const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; } return { w, h, dpr }; }
let peaksCache = {};
function peaks(buf, w, tag) {
  const k = tag + w; if (peaksCache[k]) return peaksCache[k];
  const a = buf.getChannelData(0), b = buf.getChannelData(1), n = buf.length, out = new Float32Array(w);
  for (let x = 0; x < w; x++) { const s = Math.floor(x * n / w), e = Math.floor((x + 1) * n / w); let m = 0;
    for (let i = s; i < e; i += 2) { const v = Math.max(Math.abs(a[i]), Math.abs(b[i])); if (v > m) m = v; } out[x] = m; }
  return (peaksCache[k] = out);
}
const waveCv = $('#wave'), specCv = $('#spec');
function drawWave() {
  const { w, h, dpr } = fitCanvas(waveCv); const g = waveCv.getContext('2d');
  g.fillStyle = cs['--screen']; g.fillRect(0, 0, w, h);
  const mid = h / 2;
  g.fillStyle = cs['--screen-grid']; g.fillRect(0, mid, w, Math.max(1, dpr));
  if (!track) return;
  const pO = peaks(track.buffer, w, 'o'); g.fillStyle = cs['--trace-pre'];
  for (let x = 0; x < w; x++) { const v = pO[x] * (mid - 4 * dpr); g.fillRect(x, mid - v, 1, v * 2); }
  if (master) { const pM = peaks(master.buffer, w, 'm'); g.strokeStyle = cs['--trace']; g.globalAlpha = fresh() ? 1 : 0.4; g.lineWidth = dpr;
    g.beginPath(); for (let x = 0; x < w; x++) { const y = mid - pM[x] * (mid - 4 * dpr); x ? g.lineTo(x, y) : g.moveTo(x, y); }
    for (let x = w - 1; x >= 0; x--) g.lineTo(x, mid + pM[x] * (mid - 4 * dpr)); g.closePath(); g.stroke(); g.globalAlpha = 1; }
  const px = pos() / track.buffer.duration * w;
  g.fillStyle = cs['--trace']; g.fillRect(Math.round(px), 0, Math.max(2, 2 * dpr), h);
}
$('#waveBox').addEventListener('click', e => { if (!track) return; const r = waveCv.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * track.buffer.duration); });
const FMIN = 20, FMAX = 20000, DBLO = -84, DBHI = -6;
function drawSpec() {
  const { w, h, dpr } = fitCanvas(specCv); const g = specCv.getContext('2d');
  g.fillStyle = cs['--screen']; g.fillRect(0, 0, w, h);
  const xf = f => Math.log(f / FMIN) / Math.log(FMAX / FMIN) * w, yd = d => (DBHI - d) / (DBHI - DBLO) * h;
  g.fillStyle = cs['--screen-grid']; g.font = `${10 * dpr}px ${getComputedStyle(document.body).getPropertyValue('--f-mono') || 'monospace'}`;
  [50, 100, 200, 500, 1000, 2000, 5000, 10000].forEach(f => { const x = Math.round(xf(f)); g.fillRect(x, 0, Math.max(1, dpr), h); });
  [-72, -60, -48, -36, -24, -12].forEach(d => g.fillRect(0, Math.round(yd(d)), w, Math.max(1, dpr)));
  g.fillStyle = cs['--screen-ink'];
  [[100, '100'], [1000, '1k'], [10000, '10k']].forEach(([f, t]) => g.fillText(t, xf(f) + 4 * dpr, h - 5 * dpr));
  if (!L || !playing) { g.fillText('Play to see the live spectrum', 8 * dpr, 16 * dpr); return; }
  const sr = L.ctx.sampleRate;
  const trace = (an, data, color, fill) => {
    an.getFloatFrequencyData(data); const bins = data.length, step = 2 * dpr; g.beginPath();
    for (let x = 0; x <= w; x += step) {
      const f1 = FMIN * Math.pow(FMAX / FMIN, x / w), f2 = FMIN * Math.pow(FMAX / FMIN, (x + step) / w);
      let b1 = Math.floor(f1 / (sr / 2) * bins), b2 = Math.max(b1 + 1, Math.floor(f2 / (sr / 2) * bins)); let m = -200;
      for (let b = b1; b < b2 && b < bins; b++) if (data[b] > m) m = data[b];
      const d = m + 3 * Math.log2(f1 / 1000); const y = clamp(yd(d), 0, h); x ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    if (fill) { g.lineTo(w, h); g.lineTo(0, h); g.closePath(); g.globalAlpha = 0.18; g.fillStyle = color; g.fill(); g.globalAlpha = 1; }
    else { g.strokeStyle = color; g.lineWidth = 1.25 * dpr; g.stroke(); }
  };
  trace(L.pre, L.preD, cs['--trace-pre'], false);
  trace(L.post, L.postD, cs['--trace'], true);
  trace(L.post, L.postD, cs['--trace'], false);
}
const disp = { M: -Infinity, S: -Infinity, pk: -Infinity, c: 0, l: 0 };
function drawMeters() {
  if (!playing) { meterVals.M = meterVals.S = -Infinity; meterVals.pk = 0; meterVals.c = meterVals.l = 1; }
  const lufsW = v => clamp((v + 40) / 40 * 100, 0, 100) + '%';
  disp.M = meterVals.M; disp.S = meterVals.S;
  const pkdb = lin2db(meterVals.pk); disp.pk = pkdb > disp.pk ? pkdb : disp.pk - 0.6; meterVals.pk = 0;
  const cgr = -lin2db(meterVals.c), lgr = -lin2db(meterVals.l);
  disp.c = cgr > disp.c ? cgr : Math.max(0, disp.c - 0.15); disp.l = lgr > disp.l ? lgr : Math.max(0, disp.l - 0.15);
  meterVals.c = meterVals.l = 1;
  $('#mb-m').style.width = lufsW(disp.M); $('#mv-m').textContent = num(disp.M);
  $('#mb-s').style.width = lufsW(disp.S); $('#mv-s').textContent = num(disp.S);
  $('#mb-p').style.width = lufsW(disp.pk); $('#mv-p').textContent = disp.pk < -90 ? MINUS + '∞' : num(disp.pk);
  $('#mbox-p').classList.toggle('hot', disp.pk > -0.3);
  $('#mb-c').style.width = clamp(disp.c / 12 * 100, 0, 100) + '%'; $('#mv-c').textContent = disp.c < 0.05 ? '0.0' : MINUS + disp.c.toFixed(1);
  $('#mb-l').style.width = clamp(disp.l / 12 * 100, 0, 100) + '%'; $('#mv-l').textContent = disp.l < 0.05 ? '0.0' : MINUS + disp.l.toFixed(1);
}
let lastTime = '';
function frame() {
  drawWave(); drawSpec(); drawMeters();
  if (track) { const t = fmtTime(pos()) + ' / ' + fmtTime(track.buffer.duration); if (t !== lastTime) { $('#time').textContent = t; lastTime = t; } }
  requestAnimationFrame(frame);
}

/* ---------- Boot ---------- */
syncAll(); renderReport(); requestAnimationFrame(frame);
(async () => {
  if (LOCAL && LOCAL.initial) {
    setBusy(true); status('Decoding ' + LOCAL.initial, true);
    try { await decodeAndSet(await (await api('/api/initial')).arrayBuffer(), LOCAL.initial); return; }
    catch (e) { setBusy(false); status(LOCAL.initial + ' could not be decoded here. Showing the demo loop instead.'); }
  }
  loadDemo();
})();
})();
</script>
"""


if __name__ == "__main__":
    main()
