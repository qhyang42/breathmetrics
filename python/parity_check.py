"""Parity check: the Python port vs the MATLAB reference.

1. In MATLAB, run python/make_parity_reference.m (writes python/parity_ref/).
2. Run this script:  python parity_check.py

Compares the conditioned detection trace sample-for-sample and every
landmark family (prep peaks/troughs; final onsets/peaks/troughs after
pruning). MATLAB indices are 1-based and the port is 0-based, so the
reference indices are shifted by one before comparison. Landmark agreement
is reported as exact matches and as matches within a tolerance (default
50 ms) via greedy nearest pairing, plus any unmatched counts.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zlp_breathmetrics import prep_breath_trace, find_inhale_onsets  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_ref")
TOL_S = 0.050


def load_ref():
    meta = {}
    with open(os.path.join(REF, "meta.csv")) as f:
        for row in csv.reader(f):
            meta[row[0]] = float(row[1])
    fs, n = meta["fs"], int(meta["n"])
    resp = np.fromfile(os.path.join(REF, "resp.bin"), dtype="<f8")
    det = np.fromfile(os.path.join(REF, "det.bin"), dtype="<f8")
    assert resp.size == n and det.size == n
    marks = {}
    with open(os.path.join(REF, "landmarks.csv")) as f:
        rd = csv.reader(f)
        next(rd)
        for kind, idx in rd:
            marks.setdefault(kind, []).append(int(idx) - 1)  # 1-based -> 0-based
    return fs, resp, det, {k: np.asarray(v) for k, v in marks.items()}


def compare(name, ref, got, fs, tol_s=TOL_S):
    ref = np.sort(np.asarray(ref))
    got = np.sort(np.asarray(got))
    exact = np.intersect1d(ref, got).size
    tol = int(round(tol_s * fs))
    used = np.zeros(got.size, dtype=bool)
    within = 0
    devs = []
    for r in ref:
        if got.size == 0:
            break
        j = int(np.argmin(np.where(used, np.inf, np.abs(got - r))))
        if not used[j] and abs(got[j] - r) <= tol:
            used[j] = True
            within += 1
            devs.append(abs(got[j] - r))
    dev_ms = (np.median(devs) / fs * 1000) if devs else float("nan")
    max_ms = (np.max(devs) / fs * 1000) if devs else float("nan")
    print(f"  {name:12s} ref={ref.size:4d} py={got.size:4d} exact={exact:4d} "
          f"within {tol_s*1000:.0f}ms={within:4d} "
          f"(median dev {dev_ms:.1f} ms, max {max_ms:.1f} ms) "
          f"unmatched ref={ref.size - within} py={got.size - within}")
    return ref.size == got.size and within == ref.size


def main():
    fs, resp, det_ref, marks = load_ref()
    print(f"parity: n={resp.size} @ {fs:g} Hz, tolerance {TOL_S*1000:.0f} ms")

    det, pk0, tr0, _ = prep_breath_trace(resp, fs, "conservative", None, None, 0.05)
    dd = np.abs(det - det_ref)
    print(f"  det trace    max|diff|={dd.max():.3e}  (rms {np.sqrt((dd**2).mean()):.3e})")

    ok = True
    ok &= compare("prepPeak", marks["prepPeak"], pk0, fs)
    ok &= compare("prepTrough", marks["prepTrough"], tr0, fs)

    on, pkp, trp = find_inhale_onsets(det, fs, pk0, tr0, "kneeBacktrack",
                                      0.4, 1.25, 0.50, 0.10)
    ok &= compare("onset", marks["onset"], on, fs)
    ok &= compare("peak", marks["peak"], pkp, fs)
    ok &= compare("trough", marks["trough"], trp, fs)

    print("PARITY:", "PASS" if ok else "CHECK FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
