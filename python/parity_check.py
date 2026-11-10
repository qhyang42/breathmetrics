"""Parity check: the Python port vs the MATLAB reference.

1. In MATLAB, run python/make_parity_reference.m (writes python/parity_ref/).
2. Run this script:  python parity_check.py

Two traces are checked (the toolbox's sample recording and the synthetic
trace of tests/zlpSyntheticTrace.m), each in four parts:

* the detection stage: the conditioned detection trace sample-for-sample and
  every landmark family (prep peaks/troughs; final onsets/peaks/troughs after
  pruning), as exact matches and as matches within a tolerance (default
  50 ms) via greedy nearest pairing;
* the onset refinement (``refine_inhale_onsets``), run on MATLAB's detection
  landmarks, with the default options and with three option variants: every
  refined sample, rule label and boolean flag must be identical, the real
  arrays (shifts, confidences, levels, amplitudes, ...) equal to within 1e-9
  relative;
* the incomplete-exhale merge (``merge_incomplete_exhales``), run on the
  MATLAB driver's refined paired landmarks, with the defaults and two option
  variants: every landmark, kept row, origin row, pulse count and boolean
  flag identical, the real arrays (areas, ratios, zero-flow levels,
  crossings) within 1e-9 relative;
* the driver: ``segment_breaths`` with the stages off / refinement on /
  refinement + merge against the landmarks the MATLAB driver
  ``zlpEstimateAllFeatures`` injects into the breathmetrics class.

MATLAB indices are 1-based and the port is 0-based, so reference sample
indices are shifted by one before comparison (times: by 1/fs). "Relative"
is the elementwise |py - ref| / max(|ref|, 1e-9 x max|ref| of that array),
so that entries at zero are judged against the scale of their array.
"""

import csv
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zlp_breathmetrics import (prep_breath_trace, find_inhale_onsets,  # noqa: E402
                               refine_inhale_onsets, merge_incomplete_exhales,
                               segment_breaths, _fillmissing_linear)

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity_ref")
TOL_S = 0.050
REL_TOL = 1e-9

FAILS = []
MAX_REL = {}          # stage -> largest relative difference of a real array
MAX_ABS = {}


def load_ref(case):
    d = os.path.join(REF, case)
    meta = {}
    with open(os.path.join(d, "meta.csv")) as f:
        for row in csv.reader(f):
            meta[row[0]] = float(row[1])
    fs, n = meta["fs"], int(meta["n"])
    resp = np.fromfile(os.path.join(d, "resp.bin"), dtype="<f8")
    det = np.fromfile(os.path.join(d, "det.bin"), dtype="<f8")
    assert resp.size == n and det.size == n
    marks = {}
    with open(os.path.join(d, "landmarks.csv")) as f:
        rd = csv.reader(f)
        next(rd)
        for kind, idx in rd:
            marks.setdefault(kind, []).append(int(idx) - 1)  # 1-based -> 0-based
    stages = loadmat(os.path.join(d, "stages.mat"), simplify_cells=True)
    return fs, resp, det, {k: np.asarray(v) for k, v in marks.items()}, stages


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


# ---- comparison helpers for the stage outputs ----

def _num(x):
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def _fail(stage, field, msg):
    FAILS.append((stage, field, msg))
    print(f"    FAIL {stage}: {field}: {msg}")


def exact(stage, field, got, ref, shift=0):
    """Identical integers (``shift`` is subtracted from the reference: 1 for
    MATLAB's 1-based samples / rows)."""
    g = _num(got)
    r = _num(ref) - shift
    if g.size != r.size:
        _fail(stage, field, f"length {g.size} vs {r.size}")
    elif not np.array_equal(np.nan_to_num(g, nan=-1e18), np.nan_to_num(r, nan=-1e18)):
        _fail(stage, field, f"{int(np.sum(g != r))} of {g.size} entries differ")


def flags(stage, field, got, ref):
    g = np.atleast_1d(np.asarray(got)).astype(bool)
    r = np.atleast_1d(np.asarray(ref)).astype(bool)
    if g.shape != r.shape or not np.array_equal(g, r):
        _fail(stage, field, f"{int(np.sum(g != r)) if g.shape == r.shape else 'shape'} flags differ")


def labels(stage, field, got, ref):
    # loadmat returns an empty MATLAB char as an empty array
    g = [str(s) for s in got]
    r = ["" if np.size(s) == 0 else str(s) for s in np.atleast_1d(ref)]
    if g != r:
        _fail(stage, field, f"{sum(a != b for a, b in zip(g, r))} of {len(r)} labels differ")


def real(stage, field, got, ref, shift=0.0):
    """Real arrays: NaN pattern identical, values within REL_TOL relative."""
    g = _num(got)
    r = _num(ref) - shift
    if g.size != r.size:
        _fail(stage, field, f"length {g.size} vs {r.size}")
        return
    if not np.array_equal(np.isnan(g), np.isnan(r)):
        _fail(stage, field, "NaN pattern differs")
        return
    fin = np.isfinite(g) | np.isfinite(r)
    if not fin.any():
        return
    inf_same = np.array_equal(g[np.isinf(g) | np.isinf(r)], r[np.isinf(g) | np.isinf(r)])
    g, r = g[np.isfinite(g) & np.isfinite(r)], r[np.isfinite(g) & np.isfinite(r)]
    if not inf_same:
        _fail(stage, field, "infinite entries differ")
        return
    if g.size == 0:
        return
    scale = max(np.abs(r).max(), 1e-300)
    ad = np.abs(g - r)
    rel = (ad / np.maximum(np.abs(r), 1e-9 * scale)).max()
    MAX_REL[stage] = max(MAX_REL.get(stage, 0.0), rel)
    MAX_ABS[stage] = max(MAX_ABS.get(stage, 0.0), ad.max())
    if rel > REL_TOL:
        _fail(stage, field, f"max relative diff {rel:.3e} (abs {ad.max():.3e})")


def scalar(stage, field, got, ref):
    real(stage, field, [got], [ref])


# ---- the stages ----

def stage_refine(case, fs, resp, st, eng):
    respf = _fillmissing_linear(resp)
    on, pk, tr = eng
    print(f"  refinement (unit, on MATLAB's detection landmarks)")
    on2, info = refine_inhale_onsets(respf, fs, on, pk, tr)
    ri = st["refInfo"]
    tag = f"{case}/refine"
    exact(tag, "onsets", on2, st["refOn"], 1)
    labels(tag, "rule", info["rule"], ri["rule"])
    for py, ml in (("shift_ms", "shiftMs"), ("confidence", "confidence"), ("scale_ms", "scaleMs"),
                   ("level", "level"), ("level_ref", "levelRef"), ("level_margin", "levelMargin"),
                   ("amp_local", "ampLocal"), ("amp_breath", "ampBreath"), ("level_amp_nb", "levelAmpNb"),
                   ("level_band", "levelBand")):
        real(tag, py, info[py], ri[ml])
    for py, ml in (("level_band_width", "levelBandWidth"), ("amp_session", "ampSession")):
        scalar(tag, py, info[py], ri[ml])
    for py, ml in (("level_guarded", "levelGuarded"), ("level_pause_floor", "levelPauseFloor"),
                   ("level_trough_refused", "levelTroughRefused")):
        flags(tag, py, info[py], ri[ml])
    for py, ml in (("n_foot", "nFoot"), ("n_far", "nFar"), ("n_level", "nLevel"),
                   ("n_clamp", "nClamp"), ("n_unchanged", "nUnchanged")):
        exact(tag, py, info["counts"][py], ri["counts"][ml])
    c = info["counts"]
    print(f"    {on.size} onsets: moved {on.size - c['n_unchanged']} (foot {c['n_foot']}, far {c['n_far']}, "
          f"level {c['n_level']}, clamp {c['n_clamp']}); shift median "
          f"{np.median(np.abs(info['shift_ms'][info['shift_ms'] != 0])):.0f} ms, "
          f"max {np.max(np.abs(info['shift_ms'])):.0f} ms; guarded {info['n_level_guarded']}, "
          f"pause floor {info['n_level_pause_floor']}, trough refused {info['n_level_trough_refused']}")
    camel = {"levelRefMode": "level_ref_mode", "levelGuard": "level_guard", "levelTroughFrac": "level_trough_frac"}
    for q, rv in enumerate(np.atleast_1d(st["refVar"])):
        opts = {camel[k]: v for k, v in rv["opts"].items()}
        o2, i2 = refine_inhale_onsets(respf, fs, on, pk, tr, opts)
        t2 = f"{case}/refine-variant{q + 1}"
        exact(t2, "onsets", o2, rv["on"], 1)
        labels(t2, "rule", i2["rule"], rv["rule"])
        real(t2, "shift_ms", i2["shift_ms"], rv["shiftMs"])
        real(t2, "confidence", i2["confidence"], rv["confidence"])
        real(t2, "level", i2["level"], rv["level"])


def _orig_rows(ref):
    rows = ref["origRows"]
    if isinstance(rows, np.ndarray) and rows.dtype != object and rows.ndim == 1 and np.ndim(rows[0]) == 0:
        rows = [rows]           # a single merged breath collapses to one vector
    return [np.atleast_1d(r).astype(int) - 1 for r in rows]


def check_merge(tag, got_m, got_i, ref_m, ref_i, fs):
    for py, ml in (("onsets", "onsets"), ("ends", "ends"), ("peaks", "peaks"), ("troughs", "troughs"),
                   ("kept_row", "keptRow")):
        exact(tag, py, got_m[py], ref_m[ml], 1)
    exact(tag, "n_pulses", got_m["n_pulses"], ref_m["nPulses"])
    exact(tag, "n_rows", got_m["n_rows"], ref_m["nRows"])
    g_rows, r_rows = got_m["orig_rows"], _orig_rows(ref_m)
    if len(g_rows) != len(r_rows) or any(not np.array_equal(a, b) for a, b in zip(g_rows, r_rows)):
        _fail(tag, "orig_rows", "differ")
    real(tag, "inhale_area", got_m["inhale_area"], ref_m["inhaleArea"])
    real(tag, "exhale_area", got_m["exhale_area"], ref_m["exhaleArea"])
    real(tag, "ex_in_ratio", got_m["ex_in_ratio"], ref_m["exInRatio"])
    for py, ml in (("n_before", "nBefore"), ("n_after", "nAfter"), ("n_minor", "nMinor"),
                   ("n_lead_bumps", "nLeadBumps"), ("n_merged", "nMerged"), ("n_trough_heads", "nTroughHeads")):
        exact(tag, py, got_i[py], ref_i[ml])
    for py, ml in (("is_minor", "isMinor"), ("is_lead_bump", "isLeadBump"), ("trough_left", "troughLeft"),
                   ("pause_left", "pauseLeft"), ("is_trough_head", "isTroughHead"), ("is_tail", "isTail")):
        flags(tag, py, got_i[py], ref_i[ml])
    for py, ml in (("pulse_inhale_area", "pulseInhaleArea"), ("pair_ex_in_ratio", "pairExInRatio"),
                   ("zero_flow_at_onsets", "zeroFlowAtOnsets"), ("dead_band", "deadBand"),
                   ("pair_inhale_area", "pairInhaleArea"), ("pair_exhale_area", "pairExhaleArea"),
                   ("removal_ratio", "removalRatio")):
        real(tag, py, got_i[py], ref_i[ml])
    real(tag, "engine_onset_sec", got_i["engine_onset_sec"], ref_i["engineOnsetSec"], 1.0 / fs)
    real(tag, "pair_cross_sec", got_i["pair_cross_sec"], ref_i["pairCrossSec"], 1.0 / fs)
    scalar(tag, "zero_flow_frac_dwell", got_i["zero_flow_frac_dwell"], ref_i["zeroFlowFracDwell"])
    for py, ml in (("trough_win_sec", "troughWinSec"), ("pause_win_sec", "pauseWinSec"),
                   ("pause_gap_sec", "pauseGapSec")):
        scalar(tag, "keep_opts." + py, got_i["keep_opts"][py], ref_i["keepOpts"][ml])
    for key in ("rule", "version", "zero_flow"):
        if got_i[key] != ref_i[{"rule": "rule", "version": "version", "zero_flow": "zeroFlow"}[key]]:
            _fail(tag, key, "text differs")


def stage_merge(case, fs, resp, st):
    respf = _fillmissing_linear(resp)
    on = _num(st["mergeInOn"]).astype(int) - 1
    pk = _num(st["mergeInPk"]).astype(int) - 1
    tr = _num(st["mergeInTr"]).astype(int) - 1
    print("  merge (unit, on the MATLAB driver's refined paired landmarks)")
    m, i = merge_incomplete_exhales(respf, fs, on, pk, tr)
    check_merge(f"{case}/merge", m, i, st["merged"], st["mergeInfo"], fs)
    print(f"    {i['n_before']} -> {i['n_after']} breaths: minor {i['n_minor']} "
          f"(lead bumps {i['n_lead_bumps']}), merged {i['n_merged']}, trough heads {i['n_trough_heads']}")
    camel = {"step3Mode": "step3_mode", "keepRule": "keep_rule"}
    for q, mv in enumerate(np.atleast_1d(st["mergeVar"])):
        opts = {camel[k]: v for k, v in dict(mv["spec"]["opts"]).items()}
        m2, i2 = merge_incomplete_exhales(respf, fs, on, pk, tr, float(mv["spec"]["exFrac"]), opts)
        check_merge(f"{case}/merge-variant{q + 1}", m2, i2, mv["merged"], mv["info"], fs)
        print(f"    variant {q + 1}: {i2['n_before']} -> {i2['n_after']}")


def stage_driver(case, fs, resp, st):
    print("  driver (segment_breaths landmarks vs zlpEstimateAllFeatures)")
    for name, kw, ref in (("refine off", dict(refine_onsets=False), st["driver0"]),
                          ("refine on", dict(), st["driver1"]),
                          ("refine + merge", dict(merge_incomplete_exhales=True), st["driver2"])):
        br = segment_breaths(resp, fs, **kw)
        d = br["_detection"]
        tag = f"{case}/driver[{name}]"
        exact(tag, "onsets", d["onsets"], ref["on"], 1)
        exact(tag, "peaks", d["peaks"], ref["pk"], 1)
        exact(tag, "troughs", d["troughs"], ref["tr"], 1)
        if name == "refine + merge":
            exact(tag, "n_trough_guard", br["_merge"]["n_trough_guard"], ref["nTroughGuard"])
        if name == "refine on":
            rr = br["_refine"]
            dr = st["drvRefine"]
            exact(tag, "refine.onsets_out", rr["onsets_out"], dr["onsetsOut"], 1)
            labels(tag, "refine.rule", rr["rule"], dr["rule"])
            real(tag, "refine.shift_ms", rr["shift_ms"], dr["shiftMs"])
            exact(tag, "refine.n_orphan_peaks_in", rr["n_orphan_peaks_in"], dr["nOrphanPeaksIn"])
            exact(tag, "refine.n_orphan_peaks_out", rr["n_orphan_peaks_out"], dr["nOrphanPeaksOut"])
        print(f"    {name:15s} {d['onsets'].size} onsets")


def run_case(case):
    fs, resp, det_ref, marks, st = load_ref(case)
    print(f"parity [{case}]: n={resp.size} @ {fs:g} Hz, tolerance {TOL_S*1000:.0f} ms")
    ok = True

    det, pk0, tr0, _ = prep_breath_trace(resp, fs, "conservative", None, None, 0.05)
    dd = np.abs(det - det_ref)
    print(f"  det trace    max|diff|={dd.max():.3e}  (rms {np.sqrt((dd**2).mean()):.3e})")
    ok &= compare("prepPeak", marks["prepPeak"], pk0, fs)
    ok &= compare("prepTrough", marks["prepTrough"], tr0, fs)

    on, pkp, trp = find_inhale_onsets(det, fs, pk0, tr0, "kneeBacktrack",
                                      0.4, 1.25, 0.50, 0.10)
    ok &= compare("onset", marks["onset"], on, fs)
    ok &= compare("peak", marks["peak"], pkp, fs)
    ok &= compare("trough", marks["trough"], trp, fs)

    n_before = len(FAILS)
    stage_refine(case, fs, resp, st, (on, pkp, trp))
    stage_merge(case, fs, resp, st)
    stage_driver(case, fs, resp, st)
    return ok and len(FAILS) == n_before


def main():
    ok = True
    for case in ("sample", "synth"):
        ok &= run_case(case)
        print()
    print("largest relative difference of a real array, per stage:")
    for stage in sorted(MAX_REL):
        print(f"  {stage:28s} rel {MAX_REL[stage]:.3e}   abs {MAX_ABS[stage]:.3e}")
    print("PARITY:", "PASS" if ok else "CHECK FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
