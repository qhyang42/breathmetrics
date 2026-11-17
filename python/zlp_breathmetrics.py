"""ZLP breath segmentation - Python port of the MATLAB implementation.

Port of the custom segmentation front end contributed to breathmetrics
(``breathmetrics_functions/prepBreathTrace_zlp.m`` and
``breathmetrics_functions/findInhaleOnsets_zlp.m``), of the two stages that
follow the detection (the onset refinement ``refineInhaleOnsets_zlp.m`` and
the incomplete-exhale breath merge ``mergeIncompleteExhales_zlp.m``), plus a
convenience driver. The algorithm specification lives in
``docs/zlp_segmentation.md``; this module reproduces the MATLAB functions
constant for constant, with helper routines that replicate MATLAB's
``movmean``/``movstd`` (centered, shrinking windows), directional
``movmax``/``movmin``/``movsum``, ``fillmissing``, ``findpeaks`` (prominence
first, then MATLAB's greedy-by-height minimum-distance rule), ``prctile``,
the colon operator and ``histcounts``/``conv`` semantics.

Scope
-----
Ported here: detection-trace conditioning, validity-ruled extrema, the
eligibility rules + kneeBacktrack onset placement (with the slopeGate and
changepoint comparison methods), the onset refinement, the incomplete-exhale
merge, and raw-unit amplitude/volume measures.
NOT ported: the breathmetrics feature computer (exhale onsets, pauses,
offsets, normalized volumes, shape and secondary features) - in the MATLAB
toolbox those are computed by the unmodified ``breathmetrics`` class from
the injected landmarks (see ``zlpEstimateAllFeatures.m``). The
``segment_breaths`` driver below therefore reports only measures that are
well defined from the landmarks themselves, and its inhale/exhale volume
split uses the peak as the divider (documented proxy) rather than the
toolbox's pause-aware offsets.

Conventions
-----------
* All sample indices returned by this module are 0-based (Python
  convention). The MATLAB originals are 1-based; the parity harness
  accounts for the difference.
* ``fs`` is the sampling rate in Hz; times are seconds.
* The detection trace is windowed-normalized (30-s moving std); every
  threshold in the onset functions is calibrated to that scale. Amplitude
  and volume outputs of ``segment_breaths`` are computed on the raw-unit
  trace (60-s moving-mean baseline removed).

Requires numpy and scipy (``scipy.signal.find_peaks``).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

__all__ = [
    "prep_breath_trace",
    "find_inhale_onsets",
    "refine_inhale_onsets",
    "merge_incomplete_exhales",
    "segment_breaths",
    "respiratory_volume_integrals",
]


# --------------------------------------------------------------------------
# MATLAB-semantics helpers
# --------------------------------------------------------------------------

def _mround(x):
    """MATLAB round(): half away from zero (Python's round is banker's)."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))


def _fillmissing_linear(x):
    """MATLAB fillmissing(x,'linear','EndValues','nearest')."""
    x = np.asarray(x, dtype=float).ravel().copy()
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        raise ValueError("signal contains no finite samples")
    idx = np.arange(x.size)
    # np.interp is linear inside and clamps (nearest) outside the range
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def _centered_bounds(n, k):
    """Window bounds per MATLAB movmean/movstd: for odd k = 2m+1 the window
    is [i-m, i+m]; for even k = 2m it is [i-m, i+m-1]; truncated at the
    edges (shrinking windows). Returns (lo, hi) inclusive, 0-based."""
    m_back = k // 2
    m_fwd = k - m_back - 1
    i = np.arange(n)
    lo = np.maximum(0, i - m_back)
    hi = np.minimum(n - 1, i + m_fwd)
    return lo, hi


def _movmean(x, k):
    x = np.asarray(x, dtype=float)
    n = x.size
    lo, hi = _centered_bounds(n, k)
    # cumulative sums of the de-meaned trace: a trace with a large offset
    # would otherwise lose several digits in the differences
    c0 = x.mean()
    cs = np.concatenate(([0.0], np.cumsum(x - c0)))
    cnt = hi - lo + 1
    return (cs[hi + 1] - cs[lo]) / cnt + c0


def _movstd(x, k):
    """MATLAB movstd (sample std, N-1), centered shrinking windows; a
    single-sample window reads 0."""
    x = np.asarray(x, dtype=float)
    n = x.size
    lo, hi = _centered_bounds(n, k)
    cs1 = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    cnt = (hi - lo + 1).astype(float)
    s1 = cs1[hi + 1] - cs1[lo]
    s2 = cs2[hi + 1] - cs2[lo]
    var = np.zeros(n)
    mult = cnt > 1
    var[mult] = (s2[mult] - s1[mult] ** 2 / cnt[mult]) / (cnt[mult] - 1)
    np.clip(var, 0.0, None, out=var)
    return np.sqrt(var)


def _directional_bounds(n, kb, kf):
    """MATLAB mov*(x,[kb kf]) window: [i-kb, i+kf], truncated."""
    i = np.arange(n)
    lo = np.maximum(0, i - kb)
    hi = np.minimum(n - 1, i + kf)
    return lo, hi


def _movmax_dir(x, kb, kf):
    # sliding max via a monotonic deque would be faster; n is modest and
    # correctness/parity matter more here
    x = np.asarray(x, dtype=float)
    n = x.size
    lo, hi = _directional_bounds(n, kb, kf)
    out = np.empty(n)
    for i in range(n):
        out[i] = x[lo[i]:hi[i] + 1].max()
    return out


def _movmin_dir(x, kb, kf):
    x = np.asarray(x, dtype=float)
    n = x.size
    lo, hi = _directional_bounds(n, kb, kf)
    out = np.empty(n)
    for i in range(n):
        out[i] = x[lo[i]:hi[i] + 1].min()
    return out


def _movsum_trailing(x, k):
    """MATLAB movsum(x,[k-1 0]): trailing window of length k, truncated."""
    x = np.asarray(x, dtype=float)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(x.size)
    lo = np.maximum(0, i - (k - 1))
    return cs[i + 1] - cs[lo]


def _movminmax_centered(x, k, op):
    """MATLAB movmax/movmin(x,k) with a scalar (centered) window."""
    x = np.asarray(x, dtype=float)
    n = x.size
    lo, hi = _centered_bounds(n, k)
    out = np.empty(n)
    fn = np.max if op == "max" else np.min
    for i in range(n):
        out[i] = fn(x[lo[i]:hi[i] + 1])
    return out


def _matlab_findpeaks(x, prominence, distance):
    """MATLAB findpeaks(x,'MinPeakProminence',p,'MinPeakDistance',d):
    prominence is applied FIRST, then peaks closer than d samples to a
    HIGHER surviving peak are removed (greedy by descending height).
    scipy applies distance before prominence, so the two filters are run
    manually in MATLAB's order. Returns 0-based peak indices, ascending."""
    locs, props = find_peaks(x, prominence=prominence)
    if locs.size == 0 or distance <= 1:
        return locs
    order = np.argsort(x[locs])[::-1]        # tallest first
    keep = np.ones(locs.size, dtype=bool)
    kept_locs = []
    for j in order:
        if any(abs(locs[j] - kl) < distance for kl in kept_locs):
            keep[j] = False
        else:
            kept_locs.append(locs[j])
    return np.sort(locs[keep])


def _enforce_alt(det, pk, tr, min_sep=0):
    """Strict trough->peak->trough alternation; of two same-type extrema in
    a row the more extreme wins (port of the MATLAB enforceAlt)."""
    pk = np.asarray(pk, dtype=int)
    tr = np.asarray(tr, dtype=int)
    if min_sep > 0:
        keep = np.ones(pk.size, dtype=bool)
        for k in range(1, pk.size):
            if pk[k] - pk[k - 1] < min_sep:
                keep[k] = False
        pk = pk[keep]
    ev = np.concatenate([
        np.column_stack([pk, np.ones(pk.size, dtype=int)]),
        np.column_stack([tr, -np.ones(tr.size, dtype=int)]),
    ])
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    keep = np.ones(ev.shape[0], dtype=bool)
    i = 0
    while i < ev.shape[0] - 1:
        j = i + 1
        while j < ev.shape[0] and not keep[j]:
            j += 1
        if j >= ev.shape[0]:
            break
        if ev[i, 1] == ev[j, 1]:
            if ev[i, 1] == 1:
                if det[ev[i, 0]] >= det[ev[j, 0]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    i = j
            else:
                if det[ev[i, 0]] <= det[ev[j, 0]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    i = j
        else:
            i = j
    ev = ev[keep]
    peaks = ev[ev[:, 1] == 1, 0]
    troughs = ev[ev[:, 1] == -1, 0]
    return peaks.astype(int), troughs.astype(int)


# --------------------------------------------------------------------------
# Stage 0-1: conditioning + extrema  (port of prepBreathTrace_zlp.m)
# --------------------------------------------------------------------------

def prep_breath_trace(rsp, fs, mode="conservative", blank_below_frac=None,
                      sigh_span=None, floor_frac=0.05):
    """Condition the respiration trace and find validity-ruled extrema.

    Parameters mirror ``prepBreathTrace_zlp.m`` (``sigh_span`` = the MATLAB
    ``cySpan``, a 0-based [start, end] sample pair). Returns
    ``(det, peaks, troughs, info)`` with 0-based, strictly alternating
    index arrays. ``mode='conservative'`` is the locked production choice.
    """
    rsp = _fillmissing_linear(rsp)
    base = _movmean(rsp, _mround(0.50 * fs))
    sc = _movstd(base, _mround(30 * fs))
    sc_floor = floor_frac * np.median(sc)
    x = base / np.maximum(sc, sc_floor)
    if blank_below_frac is not None:
        x = x.copy()
        x[sc < blank_below_frac * np.median(sc)] = 0.0
    n = x.size
    info = {"mode": mode, "n_breakpoints": None}

    if mode == "pwl":
        ds = max(1, _mround(fs / 20))
        gi = np.arange(0, n, ds)
        g = x[gi]
        ng = gi.size
        keep_bp = np.zeros(ng, dtype=bool)
        keep_bp[0] = keep_bp[-1] = True
        stack = [(0, ng - 1)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            seg = g[a:b + 1]
            lin = np.linspace(seg[0], seg[-1], b - a + 1)
            err = np.abs(seg - lin)
            ei = int(np.argmax(err))
            if err[ei] > 0.15:
                c = a + ei
                keep_bp[c] = True
                stack.append((a, c))
                stack.append((c, b))
        bp = gi[keep_bp]
        det = np.interp(np.arange(n), bp, x[bp])
        info["n_breakpoints"] = int(bp.size)
        v = x[bp]
        pk = [bp[k] for k in range(1, bp.size - 1)
              if v[k] > v[k - 1] and v[k] > v[k + 1]]
        tr = [bp[k] for k in range(1, bp.size - 1)
              if v[k] < v[k - 1] and v[k] < v[k + 1]]
        peaks, troughs = _enforce_alt(det, pk, tr, min_sep=_mround(1.5 * fs))

    elif mode == "conservative":
        det = x
        pk = _matlab_findpeaks(det, 0.6, _mround(1.0 * fs))
        tr = _matlab_findpeaks(-det, 0.6, _mround(1.0 * fs))
        # peak validity: HEIGHT FLOOR (+0.5) and SOFT DESCENT (0.5 drop
        # within the next 1 s), the descent waived for peaks >= 1.0 - kills
        # exhale-recovery crests before breathing pauses
        fwd_min = _movmin_dir(det, 0, _mround(1.0 * fs))
        sel = (det[pk] >= 1.0) | ((det[pk] >= 0.5) & (det[pk] - fwd_min[pk] >= 0.5))
        pk = pk[sel]
        peaks, troughs = _enforce_alt(det, pk, tr)
        # rise-fraction filter: a trough->peak rise under 40% of the local
        # breath amplitude is a bump, not a breath
        loc_amp = (_movminmax_centered(det, _mround(30 * fs), "max")
                   - _movminmax_centered(det, _mround(30 * fs), "min"))
        good = np.ones(peaks.size, dtype=bool)
        for k in range(peaks.size):
            t0 = troughs[troughs < peaks[k]]
            if t0.size == 0:
                continue
            if det[peaks[k]] - det[t0[-1]] < 0.4 * 0.5 * loc_amp[peaks[k]]:
                good[k] = False
        peaks, troughs = _enforce_alt(det, peaks[good], troughs)

    elif mode == "twoscale":
        det = x
        skel = _movmean(x, _mround(0.5 * fs))
        pk = _matlab_findpeaks(skel, 0.45, _mround(2 * fs))
        tr = _matlab_findpeaks(-skel, 0.45, _mround(2 * fs))
        w = _mround(0.6 * fs)
        pk = pk.copy()
        for k in range(pk.size):
            a = max(0, pk[k] - w); b = min(n - 1, pk[k] + w)
            pk[k] = a + int(np.argmax(det[a:b + 1]))
        tr = tr.copy()
        for k in range(tr.size):
            a = max(0, tr[k] - w); b = min(n - 1, tr[k] + w)
            tr[k] = a + int(np.argmin(det[a:b + 1]))
        peaks, troughs = _enforce_alt(det, pk, tr)

    else:
        raise ValueError(f"unknown prep mode {mode!r}")

    # paced-sigh span: of two peaks within 5 s inside the span keep the FIRST
    if sigh_span is not None and peaks.size > 1:
        keep = np.ones(peaks.size, dtype=bool)
        last = -np.inf
        for k in range(peaks.size):
            in_span = sigh_span[0] <= peaks[k] <= sigh_span[1]
            if in_span and (peaks[k] - last) < 5 * fs:
                keep[k] = False
            else:
                last = peaks[k]
        peaks, troughs = _enforce_alt(det, peaks[keep], troughs)

    return det, peaks, troughs, info


# --------------------------------------------------------------------------
# Stage 2-3: eligibility + onset placement  (port of findInhaleOnsets_zlp.m)
# --------------------------------------------------------------------------

def find_inhale_onsets(det, fs, peaks, troughs, method="kneeBacktrack",
                       r2_factor=0.4, r3_factor=1.25, dip_frac=0.50,
                       dip_dur=0.10):
    """Place one inhale onset per trough->peak pair on the detection trace.

    Port of ``findInhaleOnsets_zlp.m``; the locked production call is the
    default argument set with ``method='kneeBacktrack'``. May DELETE
    spurious trough/peak pairs - always take ``peaks``/``troughs`` from the
    outputs. All indices 0-based.
    """
    resp = np.asarray(det, dtype=float).ravel()
    peaks = list(np.asarray(peaks, dtype=int))
    troughs = np.asarray(troughs, dtype=int)
    d = _movmean(np.concatenate(([0.0], np.diff(resp))) * fs, _mround(0.12 * fs))

    # rule 1 (not descending): x(t) - x(t-0.33s) > -0.2
    lag = _mround(0.33 * fs)
    theta = 0.20
    d33 = resp - np.concatenate((np.full(lag, resp[0]), resp[:-lag]))
    elig = d33 > -theta

    # rule 3 slope-contrast field (landing refinement only, never a gate)
    lb = _mround(0.25 * fs); lf = _mround(0.5 * fs)
    d_max_w = _movmax_dir(d, lb, lf)
    d_min_w = _movmin_dir(d, lb, lf)
    r3v = d_max_w > r3_factor * np.maximum(d_min_w, 0.1)
    # legacy remap kept from the MATLAB original
    if r2_factor == 0.25:
        r2_factor = 0.4

    def pair_valid(w0, w1):
        """eligible AND rule 2 (rising ahead, pair-scaled window) over the
        inclusive window [w0, w1]."""
        seg = resp[w0:w1 + 1]
        wlen = max(_mround(0.4 * fs), _mround(0.25 * seg.size))
        rise = _movmax_dir(seg, 0, wlen) - seg
        return elig[w0:w1 + 1] & (rise > r2_factor)

    # spurious-pair pruning
    pruned = True
    while pruned:
        pruned = False
        for k in range(len(peaks)):
            t0 = troughs[troughs < peaks[k]]
            if t0.size == 0:
                continue
            t0 = t0[-1]
            if not pair_valid(t0, peaks[k]).any():
                troughs = troughs[troughs != t0]
                del peaks[k]
                pruned = True
                break
    peaks = np.asarray(peaks, dtype=int)

    onsets = []
    for k in range(peaks.size):
        pk = peaks[k]
        tr_all = troughs[troughs < pk]
        if tr_all.size == 0:
            continue
        tr = tr_all[-1]
        n_w = pk - tr + 1
        if n_w < _mround(0.2 * fs):
            continue
        seg = resp[tr:pk + 1]
        dseg = d[tr:pk + 1]
        e_seg = pair_valid(tr, pk)
        rng_ = resp[pk] - resp[tr]
        lo = resp[tr] + 0.15 * rng_
        hi = resp[pk] - 0.25 * rng_

        if method == "slopeGate":
            th = 0.20 * dseg.max()
            sustain = _mround(0.10 * fs)
            upto = n_w - sustain
            cand = np.nonzero((dseg[:upto] > th) & (seg[:upto] >= lo)
                              & (seg[:upto] <= hi) & e_seg[:upto])[0]
            o = None
            for c in cand:
                if (dseg[c:c + sustain + 1] > th * 0.5).all():
                    o = int(c)
                    break
            if o is None and cand.size:
                o = int(cand[0])

        elif method == "kneeBacktrack":
            dmax = dseg.max()
            # anchor slope scale = max slope of the SECOND HALF only
            h2 = (n_w + 1) // 2 - 1          # 0-based first index of the 2nd half
            dmax_a = dseg[h2:].max()
            pos = np.arange(n_w)
            cand = np.nonzero((dseg >= 0.7 * dmax_a) & e_seg & (pos >= h2))[0]
            if cand.size:
                im = int(cand[-1])
            else:
                cand = np.nonzero(e_seg)[0]
                if cand.size:
                    im = int(cand[-1])
                else:
                    cand = np.nonzero(dseg >= 0.7 * dmax_a)[0]
                    im = int(cand[-1]) if cand.size else int(np.argmax(dseg))
            sus_len = max(1, _mround(dip_dur * fs))
            sus_dip = _movsum_trailing((dseg < dip_frac * dmax).astype(float),
                                       sus_len) >= sus_len
            o = im
            while o > 0 and not sus_dip[o - 1]:
                o -= 1
            if seg[o] < resp[tr] + 0.1 * rng_:
                # clean sweep: last upward crossing of the midpoint
                mid = resp[tr] + 0.5 * rng_
                cr = np.nonzero((seg[:-1] < mid) & (seg[1:] >= mid))[0]
                if cr.size:
                    o = int(cr[-1]) + 1
            else:
                if seg[o] > resp[tr] + 0.35 * rng_:
                    sus_len2 = max(1, _mround(0.15 * fs))
                    sus_dip2 = _movsum_trailing((dseg < 0.05 * dmax).astype(float),
                                                sus_len2) >= sus_len2
                    o2 = o
                    while o2 > 0 and not sus_dip2[o2 - 1]:
                        o2 -= 1
                    if o2 > 0:
                        o = o2
                r3seg = r3v[tr:pk + 1]
                if not r3seg[min(o, n_w - 1)] and r3seg.any():
                    cnd = np.nonzero(r3seg)[0]
                    o = int(cnd[np.argmin(np.abs(cnd - o))])

        elif method == "changepoint":
            nn = n_w
            step = max(1, _mround(fs / 100))
            cands = range(_mround(0.05 * nn), _mround(0.95 * nn) + 1, step)
            xx = np.arange(1, nn + 1, dtype=float)
            best = np.inf
            o = None
            for c in cands:
                if c < 2 or c > nn - 1:
                    continue
                xl, yl = xx[:c], seg[:c]
                xr, yr = xx[c - 1:], seg[c - 1:]
                pl = np.polyfit(xl, yl, 1)
                pr = np.polyfit(xr, yr, 1)
                if pr[0] <= pl[0] + 1e-9:
                    continue
                r = (np.sum((yl - np.polyval(pl, xl)) ** 2)
                     + np.sum((yr - np.polyval(pr, xr)) ** 2))
                if r < best:
                    best = r
                    o = c - 1
        else:
            raise ValueError(f"unknown method {method!r}")

        # ineligible landing -> nearest eligible sample (either direction)
        if o is not None and not e_seg[min(o, n_w - 1)]:
            cand = np.nonzero(e_seg)[0]
            if cand.size == 0:
                o = None
            else:
                o = int(cand[np.argmin(np.abs(cand - o))])
        # HARD FLOOR: never in the first 20% of the trough->peak interval;
        # forward-only eligibility snap from the floor
        if o is not None:
            floor_idx = int(np.ceil(0.20 * (n_w - 1)))
            if o < floor_idx:
                cand = np.nonzero(e_seg[floor_idx:])[0]
                o = floor_idx + int(cand[0]) if cand.size else floor_idx
        if o is not None:
            onsets.append(tr + o)

    return np.asarray(sorted(onsets), dtype=int), peaks, troughs


# --------------------------------------------------------------------------
# MATLAB-semantics helpers for the refinement / merge ports
# --------------------------------------------------------------------------

def _pad1(a):
    """Prepend a dummy element so that ``out[i]`` is MATLAB's ``a(i)``
    (1-based). The refinement and merge routines are transcribed from
    MATLAB with 1-based sample numbers; only their inputs and outputs are
    converted to the 0-based convention of this module."""
    return np.concatenate(([np.nan], np.asarray(a, dtype=float)))


def _seg1(a1, lo, hi):
    """MATLAB ``a(lo:hi)`` on a 1-padded array (inclusive, ``lo >= 1``)."""
    if lo < 1:
        raise IndexError(f"index {lo} is outside the trace (1-based)")
    return a1[lo:hi + 1]


def _matlab_colon(a, d, b):
    """MATLAB ``a:d:b`` for doubles with ``d > 0``. numpy.arange differs in
    the last bits and, near an integer number of steps, in the length; the
    histogram bin edges of the zero-flow search are built with this."""
    eps = np.finfo(float).eps
    ndbl = np.floor((b - a) / d + 0.5)
    apnd = a + ndbl * d
    cdiff = apnd - b
    if abs(cdiff) < 2.0 * eps * max(abs(a), abs(b)):
        ndbl += 1
        apnd = b
    elif cdiff > 0:
        apnd = a + (ndbl - 1.0) * d
    else:
        ndbl += 1
    n = int(ndbl) if ndbl >= 0 else 0
    y = np.empty(n)
    if n > 0:
        y[0] = a
        if n > 1:
            y[n - 1] = apnd
            half = (n - 1) >> 1
            for k in range(half - 1):
                kd = (k + 1) * d
                y[k + 1] = a + kd
                y[n - 2 - k] = apnd - kd
            if (half << 1) == n - 1:
                y[half] = (a + apnd) / 2.0
            else:
                kd = half * d
                y[half] = a + kd
                y[half + 1] = apnd - kd
    return y


def _matlab_prctile(w, pcts):
    """MATLAB prctile of a vector without NaNs (midpoint rule: the i-th
    sorted value is the (i - 0.5)/n quantile, linear in between)."""
    s = np.sort(np.asarray(w, dtype=float))
    n = s.size
    out = []
    for p in pcts:
        r = p / 100.0 * n
        if r < 0.5:
            out.append(s[0])
        elif r > n - 0.5:
            out.append(s[-1])
        else:
            k = int(np.floor(r + 0.5))
            t = r + 0.5 - k
            out.append((1 - t) * s[k - 1] + t * s[min(k, n - 1)])
    return out


def _fma(a, b, c):
    """a*b + c with a single rounding (math.fma needs Python 3.13)."""
    from fractions import Fraction
    return float(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def _conv3_same(c):
    """MATLAB ``conv(c, [1 1 1]/3, 'same')`` for a vector of counts.

    The values are only used to pick the highest bin, so ties between equal
    integer sums matter and depend on the floating-point accumulation order.
    MATLAB's order, as measured against R2026a on random count vectors: up to
    32 elements ``(c[i+1]*k + c[i]*k) + c[i-1]*k``; above that, the
    first ``8*((n-1)//8) + 1`` outputs are accumulated with fused
    multiply-adds, ``fma(c[i-1], k, fma(c[i], k, c[i+1]*k))``, and the
    rest as before."""
    c = np.asarray(c, dtype=float)
    n = c.size
    k = 1.0 / 3.0
    z = np.zeros(n + 2)
    z[1:-1] = c
    out = (z[1:-1] * k + z[2:] * k) + z[:-2] * k
    if n > 32:
        for j in range(8 * ((n - 1) // 8) + 1):
            out[j] = _fma(z[j], k, _fma(z[j + 1], k, z[j + 2] * k))
    return out


def _interp1_linear_extrap(xg, yg, xq):
    """MATLAB interp1(xg, yg, xq, 'linear', 'extrap') for sorted xg."""
    j = np.clip(np.searchsorted(xg, xq, side="right") - 1, 0, xg.size - 2)
    slope = (yg[j + 1] - yg[j]) / (xg[j + 1] - xg[j])
    return yg[j] + (xq - xg[j]) * slope


def _movmedian_shrink(v, k):
    """MATLAB movmedian(v, k) for odd k with shrinking end windows."""
    v = np.asarray(v, dtype=float)
    h = k // 2
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(v.size)])


def _local_median(v, w, include_self):
    """Median of the finite values within +-w entries (the entry itself
    excluded unless ``include_self``); NaN when none."""
    n = v.size
    m = np.full(n, np.nan)
    for k in range(n):
        a = max(0, k - w)
        b = min(n - 1, k + w)
        win = v[a:b + 1].copy()
        if not include_self:
            win[k - a] = np.nan
        win = win[np.isfinite(win)]
        if win.size:
            m[k] = np.median(win)
    return m


def _median_finite(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return 0.0 if a.size == 0 else float(np.median(a))


def _nanmedian(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return np.nan if a.size == 0 else float(np.median(a))


def _check_options(user, defaults, who):
    o = dict(defaults)
    if user is not None:
        for key, val in dict(user).items():
            if key not in o:
                raise ValueError(f"{who}: unknown option {key!r}")
            o[key] = val
    return o


# --------------------------------------------------------------------------
# Stage 4: onset refinement  (port of refineInhaleOnsets_zlp.m)
# --------------------------------------------------------------------------

REFINE_VERSION = "refineInhaleOnsets_zlp 2026-10-29a"

_REFINE_DEFAULTS = dict(
    level_smooth_ms=40,                 # levels are read from this moving mean
    scales_ms=(40, 80, 160, 300),       # slope scales (moving mean + central-difference span)
    snr_min=5,                          # a scale is usable when its steepest slope >= snr_min x local slope noise
    noise_win_sec=10, noise_sigmas=2,   # noise window (+- s); foot slope threshold >= noise_sigmas x noise
    bias_frac=0.2,                      # foot moved forward by bias_frac x scale (moving-mean footprint)
    back_ms=200,                        # rule 1 searches from onset - back_ms ...
    min_rise_ms=100,                    # ... to peak - min_rise_ms
    guard_ms=50,                        # never closer than this after the previous peak
    trough_tol_ms=0,                    # never earlier than the pair's trough minus this
    foot_frac=0.05, dwell_ms=50,        # foot: slope < foot_frac x steepest for dwell_ms
    far_tau_sec=2.5,                    # proximity weight exp(-(p - t)/far_tau_sec) of the steepest rise
    far_mult=1.5, far_min_sec=0.8, far_skip_frac=0.40,    # rule 2
    min_shift_ms=20,                    # smaller moves are not made
    skip_rise_frac=0.15,                # a forward move may skip at most this x amplitude of level ...
    skip_slope_frac=0.45,               # ... and at most skip_steep_frac x amplitude when the skipped stretch is steep
    skip_steep_frac=0.05,
    earlier_rise_frac=0.25,             # an earlier rise counts when its slope >= this x the steepest
    level_win_breaths=8,                # rule 2: neighbours each side for the median rise time
    level_band_frac=0.30, level_gain_frac=0.10, low_margin_frac=1.0,   # rule 3 (whole-trace band)
    level_ref_mode="session", level_margin_frac=0.15,     # 'local' = neighbour reference of an earlier version
    level_guard="pause", level_guard_sec=0.5,             # rule 3 guard: 'pause' | 'rise' | 'none'
    level_rise_frac=0.10, level_pause_frac=0.15, level_pause_level_frac=0.25,
    level_trough_frac=0.35, level_trough_win_sec=1.0,
)


class _RefineCtx:
    """The traces and options shared by the foot searches of one call.
    ``s`` and the entries of ``D`` are 1-padded (see ``_pad1``)."""

    def __init__(self, s, D, fs, o, dwell, NZ=None):
        self.s, self.D, self.fs, self.o, self.dwell, self.NZ = s, D, fs, o, dwell, NZ


def _foot_multi(ctx, k, lo, hi, p):
    """Foot of the steepest (proximity-weighted) rise in [lo, hi] at the
    finest usable slope scale. Returns (foot, slope, scale index); NaN /
    None = none."""
    o, fs = ctx.o, ctx.fs
    if hi < lo:
        return np.nan, np.nan, None
    seg = np.arange(lo, hi + 1)
    w = np.exp(-np.maximum(0, p - seg) / fs / o["far_tau_sec"])
    for q in range(len(o["scales_ms"])):
        dq = ctx.D[q]
        nz = ctx.NZ[k, q]
        vals = dq[lo:hi + 1] * w
        j = int(np.argmax(vals))
        if not vals[j] > 0:
            continue
        m = lo + j
        dmq = dq[m]
        if dmq < o["snr_min"] * nz:
            continue
        thr = np.fmax(o["foot_frac"] * dmq, o["noise_sigmas"] * nz)
        low = dq[lo:m + 1] < thr
        ok_run = _movsum_trailing(low.astype(float), ctx.dwell) == ctx.dwell
        ok_run[:min(ok_run.size, ctx.dwell - 1)] = False
        idx = np.nonzero(ok_run)[0]
        if idx.size == 0:
            continue
        f = lo + int(idx[-1])
        f = min(m - 1, f + _mround(o["bias_frac"] * o["scales_ms"][q] / 1000 * fs))
        return f, dmq, q
    return np.nan, np.nan, None


def _skipped_slope(d, a, b):
    """90th percentile (nearest rank) of the slope over [a, b)."""
    if b <= a:
        return -np.inf
    w = np.sort(d[a:b])
    return w[min(w.size, max(1, int(np.ceil(0.9 * w.size)))) - 1]


def _foot_checked(ctx, k, lo, hi, p, ok, skip_frac, Ak):
    """Foot of the steepest rise; a forward foot that skips more than
    ``skip_frac`` x Ak of level above the onset ``ok`` (or more than
    skip_steep_frac x Ak over a steep stretch) is replaced by the foot of an
    earlier rise (slope >= earlier_rise_frac of the steepest), up to 3
    times, else NaN (keep the detection's onset)."""
    o, s = ctx.o, ctx.s
    f, dm, sq = _foot_multi(ctx, k, lo, hi, p)
    if not np.isfinite(f):
        return f, dm, sq
    dm0 = dm
    tries = 0
    while f > ok and ((s[f] - s[ok]) > skip_frac * Ak or
                      ((s[f] - s[ok]) > o["skip_steep_frac"] * Ak and
                       _skipped_slope(ctx.D[sq], ok, f) > o["skip_slope_frac"] * dm)):
        e = f - ctx.dwell
        if tries >= 3 or e <= lo + 2 * ctx.dwell:
            return np.nan, np.nan, None
        f2, d2, q2 = _foot_multi(ctx, k, lo, e, p)
        if not np.isfinite(f2) or d2 < o["earlier_rise_frac"] * dm0:
            return np.nan, np.nan, None
        f, dm, sq = f2, d2, q2
        tries += 1
    return f, dm, sq


def _foot_level(ctx, k, lo, hi, p, thr_lvl, low_lvl, A):
    """Rule 3: foot of the steepest rise; while above ``thr_lvl`` the foot of
    the earlier rise (never below ``low_lvl``); else the last level crossing
    at or below ``thr_lvl``, refined to the foot of the rise there."""
    s, o, fs, dwell = ctx.s, ctx.o, ctx.fs, ctx.dwell
    f, dm, sq = np.nan, np.nan, None
    dm0 = np.nan
    e = hi
    for it in range(1, 6):
        f1, d1, q1 = _foot_multi(ctx, k, lo, e, p)
        if not np.isfinite(f1) or s[f1] < low_lvl:
            break
        if it == 1:
            dm0 = d1
        f, dm, sq = f1, d1, q1
        if s[f] <= thr_lvl:
            return f, dm, sq
        e = f1 - dwell
        if e <= lo + 2 * dwell:
            break
        if np.max(ctx.D[q1][lo:e + 1]) < o["earlier_rise_frac"] * dm0:
            break
    if not np.isfinite(f):
        f, dm, sq = hi, np.nan, None
    win = s[lo:f + 1]
    hit = np.nonzero((win <= thr_lvl) & (win >= low_lvl))[0]
    if hit.size:
        c = lo + int(hit[-1])
        f2, d2, q2 = _foot_multi(ctx, k, lo, min(f, c + _mround(0.15 * fs)), p)
        if np.isfinite(f2) and f2 <= c and s[f2] >= low_lvl:
            f, dm, sq = f2, d2, q2
        else:
            f, dm, sq = c, np.nan, None
    else:
        j = int(np.argmin(win))
        vmin = win[j]
        if vmin >= low_lvl and vmin < s[f] - 0.05 * max(A, np.finfo(float).eps):
            f, dm, sq = lo + j, np.nan, None
    return f, dm, sq


def _confidence_of(s, d, f, p, dm, A):
    if not np.isfinite(dm) or dm <= 0:
        return 0.0
    fr = min(1.0, max(0.0, (s[p] - s[f]) / max(A, np.finfo(float).eps)))
    fl = min(1.0, max(0.0, 1 - max(d[f], 0.0) / dm))
    return fr * fl


def _middle_range(x, frac):
    """Width of the middle ``frac`` of the values of x (0.30: 65th minus
    35th percentile; nearest rank)."""
    v = np.sort(x[np.isfinite(x)])
    if v.size == 0:
        return 0.0
    n = v.size
    lo = v[min(n, max(1, int(np.ceil((0.5 - frac / 2) * n)))) - 1]
    hi = v[min(n, max(1, int(np.ceil((0.5 + frac / 2) * n)))) - 1]
    return float(hi - lo)


def refine_inhale_onsets(rsp_stored, fs, onsets, peaks, troughs, opts=None):
    """Re-place each inhale onset on the stored respiration trace.

    Port of ``refineInhaleOnsets_zlp.m`` (see ``docs/zlp_segmentation.md``
    for the rules). The detection finds each onset on a 500-ms moving-mean,
    30-s-normalized copy of the trace; on the stored trace that onset is
    often early (the rise starts 0.1-0.3 s later), sits on a small bump
    ahead of a quick rise, or rides high on a larger inhale. An onset moves
    only when the trace shows a definitively better one (rules ``'foot'``,
    ``'far'``, ``'level'``); peaks, troughs and the number of breaths are
    not changed.

    Parameters
    ----------
    rsp_stored : the trace the detection got (raw units, inhale positive;
        non-finite samples are filled by linear interpolation).
    fs : sampling rate (Hz).
    onsets, peaks, troughs : the detection's landmarks, 0-based samples.
    opts : dict of option overrides (snake_case names of
        ``REFINE_DEFAULTS``; unknown names raise ``ValueError``).

    Returns
    -------
    ``(onsets2, info)``: ``onsets2`` - int array, one refined 0-based onset
    per input onset; ``info`` - dict aligned with it: ``src_idx``, ``rule``
    (list of str: '' unchanged | 'foot' | 'far' | 'level', + '+clamp'),
    ``shift_ms`` (refined - detected, ms), ``confidence``, ``scale_ms``,
    ``level`` (40-ms-smoothed trace at the onset), ``level_ref``,
    ``level_margin``, ``amp_local``, ``level_band`` and ``level_band_width``,
    ``amp_session``, ``amp_breath``, ``level_amp_nb``, ``level_guarded``,
    ``level_pause_floor``, ``level_trough_refused`` (+ their ``n_`` counts),
    ``counts`` (``n_foot, n_far, n_level, n_clamp, n_unchanged``), ``opts``
    and ``version``.
    """
    o = _check_options(opts, _REFINE_DEFAULTS, "refine_inhale_onsets")
    eps = np.finfo(float).eps
    x = np.asarray(rsp_stored, dtype=float).ravel().copy()
    if not np.all(np.isfinite(x)):
        good = np.isfinite(x)
        idx = np.arange(1, x.size + 1, dtype=float)
        if good.sum() >= 2:
            x[~good] = _interp1_linear_extrap(idx[good], x[good], idx[~good])
        else:
            x[~good] = 0.0
    N = x.size
    # 1-based samples inside; converted back at the end
    on = np.asarray(onsets, dtype=float).ravel() + 1
    pk = np.asarray(peaks, dtype=float).ravel() + 1
    tr = np.asarray(troughs, dtype=float).ravel() + 1
    nOn = on.size
    on2 = on.copy()
    rule = [""] * nOn
    conf = np.zeros(nOn)
    scUsed = np.full(nOn, np.nan)
    if nOn == 0:
        info = _refine_info(np.arange(0), rule, np.zeros(0), conf, scUsed,
                            None, None, None, None, o)
        return np.zeros(0, dtype=int), info

    # ---- traces: lightly smoothed levels and multi-scale slopes ----
    s = _pad1(_movmean(x, max(1, _mround(o["level_smooth_ms"] / 1000 * fs))))
    nSc = len(o["scales_ms"])
    D = []
    HF = []
    for q in range(nSc):
        wq = max(1, _mround(o["scales_ms"][q] / 1000 * fs))
        hq = max(1, _mround(o["scales_ms"][q] / 2000 * fs))
        sq = _movmean(x, wq)
        dq = np.zeros(N)
        dq[hq:N - hq] = (sq[2 * hq:] - sq[:N - 2 * hq]) / (2 * hq / fs)
        D.append(_pad1(dq))
        HF.append(_pad1(np.abs(dq - _movmean(dq, max(3, _mround(0.5 * fs))))))
    dwell = max(2, _mround(o["dwell_ms"] / 1000 * fs))
    minRise = max(2, _mround(o["min_rise_ms"] / 1000 * fs))
    back = _mround(o["back_ms"] / 1000 * fs)
    guard = _mround(o["guard_ms"] / 1000 * fs)
    tol = _mround(o["trough_tol_ms"] / 1000 * fs)
    minShift = o["min_shift_ms"] / 1000 * fs
    ctx = _RefineCtx(s, D, fs, o, dwell)

    # ---- per-breath structure ----
    P = np.full(nOn, np.nan)
    PP = np.full(nOn, np.nan)
    T = np.full(nOn, np.nan)
    LO = np.full(nOn, np.nan)
    for k in range(nOn):
        q = pk[pk > on[k]]
        if q.size == 0:
            continue
        P[k] = q[0]
        r = pk[pk < on[k]]
        PP[k] = 1 if r.size == 0 else r[-1]
        u = tr[(tr < on[k]) & (tr > PP[k])]
        if u.size:
            T[k] = u[-1]
        lo = PP[k] + guard
        if np.isfinite(T[k]):
            lo = max(lo, T[k] - tol)
        LO[k] = max(1, min(lo, on[k]))
    valid = np.isfinite(P) & (on >= 1) & (on <= N)
    NZ = np.full((nOn, nSc), np.nan)
    nw = _mround(o["noise_win_sec"] * fs)
    for k in np.nonzero(valid)[0]:
        a = max(1, _mround(on[k]) - nw)
        b = min(N, _mround(on[k]) + nw)
        for q in range(nSc):
            NZ[k, q] = 1.4826 * np.median(HF[q][a:b + 1:5])
    ctx.NZ = NZ
    A = np.full(nOn, np.nan)
    for k in np.nonzero(valid)[0]:
        A[k] = s[_mround(P[k])] - np.min(_seg1(s, _mround(LO[k]), _mround(on[k])))

    # ---- rule 1 'foot': foot of the steepest rise within [onset - back_ms, peak - min_rise] ----
    for k in np.nonzero(valid)[0]:
        ok = _mround(on[k])
        p = _mround(P[k])
        lo = max(_mround(LO[k]), ok - back)
        hi = p - minRise
        if hi <= lo + dwell:
            continue
        f, dm, sq = _foot_checked(ctx, k, lo, hi, p, ok, o["skip_rise_frac"], max(A[k], eps))
        if not np.isfinite(f) or abs(f - ok) < minShift:
            continue
        on2[k] = f
        rule[k] = "foot"
        scUsed[k] = o["scales_ms"][sq]
        conf[k] = _confidence_of(s, D[sq], f, p, dm, A[k])

    # ---- rule 2 'far': onsets far from their peak compared with the neighbours ----
    rise = np.full(nOn, np.nan)
    for k in np.nonzero(valid)[0]:
        rise[k] = (P[k] - on2[k]) / fs
    riseRef = _local_median(rise, o["level_win_breaths"], False)
    with np.errstate(invalid="ignore"):
        far = valid & (rise > np.fmax(o["far_min_sec"], o["far_mult"] * riseRef))
    for k in np.nonzero(far)[0]:
        ok = _mround(on2[k])
        p = _mround(P[k])
        lo = _mround(LO[k])
        if np.isfinite(riseRef[k]):
            lo = max(lo, _mround(p - o["far_mult"] * riseRef[k] * fs))
        hi = p - minRise
        if hi <= lo + dwell or lo <= ok:
            continue
        f, dm, sq = _foot_checked(ctx, k, lo, hi, p, ok, o["far_skip_frac"], max(A[k], eps))
        if not np.isfinite(f) or f - ok < minShift:
            continue
        on2[k] = f
        rule[k] = "far"
        scUsed[k] = o["scales_ms"][sq]
        conf[k] = _confidence_of(s, D[sq], f, p, dm, A[k])

    # ---- rule 3 'level': onsets riding high on a larger inhale ----
    lvl = np.full(nOn, np.nan)
    for k in np.nonzero(valid)[0]:
        lvl[k] = s[_mround(on2[k])]
    medOn = _median_finite(lvl[valid])
    bandW = _middle_range(x, o["level_band_frac"])
    Asess = _median_finite(A)
    if o["level_ref_mode"] == "local":
        Aloc = _local_median(A, o["level_win_breaths"], True)
        Aloc[~np.isfinite(Aloc)] = Asess
        lvlRef = _local_median(lvl, o["level_win_breaths"], False)
        margin = o["level_margin_frac"] * Aloc
    else:
        if o["level_ref_mode"] != "session":
            raise ValueError("level_ref_mode must be 'session' or 'local'")
        lvlRef = np.full(nOn, medOn)
        margin = np.full(nOn, bandW / 2)
        Aloc = np.full(nOn, Asess)
    gWin = max(1, _mround(o["level_guard_sec"] * fs))
    guarded = np.zeros(nOn, dtype=bool)
    troughRefused = np.zeros(nOn, dtype=bool)
    pauseFloor = np.zeros(nOn, dtype=bool)
    tWin = max(1, _mround(o["level_trough_win_sec"] * fs))
    AlocNb = _local_median(A, o["level_win_breaths"], True)
    AlocNb[~np.isfinite(AlocNb)] = Asess      # breath size near the onset (pause guard)
    with np.errstate(invalid="ignore"):
        cand3 = valid & (lvl > lvlRef + margin)
    for k in np.nonzero(cand3)[0]:
        ok = _mround(on2[k])
        p = _mround(P[k])
        lo = _mround(LO[k])
        hi = p - minRise
        if o["level_guard"] == "pause":
            we = _seg1(s, max(1, _mround(on[k]) - gWin), _mround(on[k]))
            if (np.max(we) - np.min(we) <= o["level_pause_frac"] * AlocNb[k] and
                    abs(np.mean(we) - medOn) <= o["level_pause_level_frac"] * AlocNb[k]):
                lo = max(lo, _mround(on[k]))
                pauseFloor[k] = True
        if hi <= lo + dwell:
            continue
        wg = _seg1(s, max(1, ok - gWin), ok)
        if o["level_guard"] == "rise":
            if s[ok] - np.min(wg) <= o["level_rise_frac"] * Aloc[k]:
                guarded[k] = True
                continue
        elif o["level_guard"] == "pause":
            if (np.max(wg) - np.min(wg) <= o["level_pause_frac"] * AlocNb[k] and
                    abs(np.mean(wg) - medOn) <= o["level_pause_level_frac"] * AlocNb[k]):
                guarded[k] = True
                continue
        elif o["level_guard"] != "none":
            raise ValueError("level_guard must be 'rise', 'pause' or 'none'")
        thrLvl = lvlRef[k] + margin[k]
        lowLvl = lvlRef[k] - o["low_margin_frac"] * Aloc[k]
        f, dm, sq = _foot_level(ctx, k, lo, hi, p, thrLvl, lowLvl, A[k])
        if not np.isfinite(f):
            continue
        if (o["level_trough_frac"] > 0 and
                medOn - np.min(_seg1(s, max(1, f - tWin), f)) >= o["level_trough_frac"] * AlocNb[k]):
            troughRefused[k] = True       # the new onset would come out of an exhale trough: a rebound
            continue
        if s[f] < s[ok] - o["level_gain_frac"] * max(A[k], eps) and abs(f - ok) >= minShift:
            on2[k] = f
            rule[k] = "level"
            lvl[k] = s[f]
            if sq is not None:
                scUsed[k] = o["scales_ms"][sq]
                conf[k] = _confidence_of(s, D[sq], f, p, dm, A[k])
            else:
                scUsed[k] = np.nan
                conf[k] = 0.5

    # ---- alternation: strictly between the previous peak and the own peak, strictly ascending ----
    lastOn = -np.inf
    for k in range(nOn):
        lowB = max(lastOn + 1, 1)
        if np.isfinite(PP[k]):
            lowB = max(lowB, _mround(PP[k]) + 1)
        upB = P[k] - 1 if np.isfinite(P[k]) else N
        if on2[k] < lowB or on2[k] > upB:
            on2[k] = min(max(on2[k], lowB), upB)
            if _mround(on2[k]) == _mround(on[k]):
                rule[k] = ""
                conf[k] = 0.0
            else:
                rule[k] = rule[k] + "+clamp"
        lastOn = on2[k]
    on2 = np.array([_mround(v) for v in on2], dtype=float)
    for k in range(1, nOn):
        if on2[k] <= on2[k - 1]:
            on2[k] = _mround(on[k])
            rule[k] = ""
            conf[k] = 0.0
    unchanged = on2 == np.array([_mround(v) for v in on], dtype=float)
    for k in np.nonzero(unchanged)[0]:
        rule[k] = ""
    conf[unchanged] = 0.0
    scUsed[unchanged] = np.nan
    shiftMs = (on2 - on) / fs * 1000
    info = _refine_info(np.arange(nOn), rule, shiftMs, conf, scUsed, lvl, lvlRef, margin, Aloc, o)
    info["level_band"] = np.array([medOn - bandW / 2, medOn + bandW / 2])
    info["level_guarded"] = guarded
    info["n_level_guarded"] = int(guarded.sum())
    info["level_amp_nb"] = AlocNb
    info["amp_breath"] = A
    info["level_pause_floor"] = pauseFloor
    info["n_level_pause_floor"] = int(pauseFloor.sum())
    info["level_trough_refused"] = troughRefused
    info["n_level_trough_refused"] = int(troughRefused.sum())
    info["level_band_width"] = bandW
    info["amp_session"] = Asess
    return (on2 - 1).astype(int), info


def _refine_info(src, rule, shift_ms, conf, sc_used, lvl, lvl_ref, margin, a_loc, o):
    counts = dict(
        n_foot=sum(r.startswith("foot") for r in rule),
        n_far=sum(r.startswith("far") for r in rule),
        n_level=sum(r.startswith("level") for r in rule),
        n_clamp=sum("clamp" in r for r in rule),
        n_unchanged=sum(r == "" for r in rule))
    return dict(src_idx=src, rule=list(rule), shift_ms=shift_ms, confidence=conf,
                scale_ms=sc_used, level=lvl, level_ref=lvl_ref, level_margin=margin,
                amp_local=a_loc, counts=counts, opts=dict(o), version=REFINE_VERSION)


# --------------------------------------------------------------------------
# Stage 5: incomplete-exhale merge  (port of mergeIncompleteExhales_zlp.m)
# --------------------------------------------------------------------------

MERGE_VERSION = "mergeIncompleteExhales 2026-10-29a"

_MERGE_DEFAULTS = dict(
    minor_frac=0.10, dead_frac=0.05, blip_smooth_sec=1,           # step 1
    lead_frac=0.5, rest_sec=3,                                     # step 2
    trough_win_sec=1, trough_frac=0.25, pause_win_sec=0.5,         # keep rule
    pause_frac=0.15, pause_level_frac=0.25, pause_gap_sec=0.1,
    zero_half_win_sec=10, zero_bin_frac=0.04, zero_side_frac=0.10,  # zero-flow level
    zero_mass_frac=0.15, zero_max_mean=True,
    step3_mode="pairCross",      # measurement only: 'window' = areas over the whole window
    keep_rule="troughPause",     # measurement only: 'first' = keep-first only
)


def _zero_flow_level(x, fs, half_win_sec, bin_frac, side_frac, use_max, mass_frac):
    """Zero-flow level of a nasal-pressure trace = its dwell level: within
    +-``half_win_sec`` the most frequent level of the 200-ms mean (histogram
    mode at 25 Hz), accepted only when the trace spends >= ``side_frac`` of
    the window both above and below it and >= ``mass_frac`` within one bin
    of it; elsewhere the 60-s moving mean. The higher of the two is used
    when ``use_max``. Returns ``(z, method text, fraction of the 1-s grid
    where the dwell level was accepted)``; ``z`` is 0-based, one value per
    sample."""
    N = x.size
    xs = _movmean(x, max(1, _mround(0.2 * fs)))
    dec = max(1, _mround(fs / 25))
    xd = xs[::dec]
    fd = fs / dec
    Nd = xd.size
    g = _matlab_colon(1.0, fs, float(N))               # 1-s grid, 1-based samples
    gd = np.minimum(Nd, np.array([_mround((v - 1) / dec) for v in g]) + 1)
    hw = _mround(half_win_sec * fd)
    zg = _movmean(x, _mround(60 * fs))[g.astype(int) - 1]
    ok = np.zeros(g.size, dtype=bool)
    zm = np.full(g.size, np.nan)
    for i in range(g.size):
        w = xd[max(1, gd[i] - hw) - 1:min(Nd, gd[i] + hw)]
        q5, q95 = _matlab_prctile(w, (5, 95))
        rg = q95 - q5
        if not rg > 0:
            continue
        bw = bin_frac * rg
        edges = _matlab_colon(w.min() - bw, bw, w.max() + bw)
        if edges.size < 4:
            continue
        c = _conv3_same(np.histogram(w, edges)[0].astype(float))
        im = int(np.argmax(c))
        m = (edges[im] + edges[im + 1]) / 2
        m = np.median(w[np.abs(w - m) <= bw])           # refine within the peak (no bin-grid bias)
        up = np.mean(w > m + 2 * bw)
        dn = np.mean(w < m - 2 * bw)
        mass = np.mean(np.abs(w - m) <= bw)
        if up >= side_frac and dn >= side_frac and mass >= mass_frac:
            ok[i] = True
            zm[i] = m
    if ok.any():
        zf = zm.copy()
        zf[ok] = _movmedian_shrink(zm[ok], 5)           # over 5 accepted seconds
        zg[ok] = np.maximum(zf[ok], zg[ok]) if use_max else zf[ok]
    if g.size < 2:
        raise ValueError("the recording is too short for the zero-flow level (needs two 1-s grid points)")
    z = _interp1_linear_extrap(g, zg, np.arange(1.0, N + 1))
    method = ("dwell level (histogram mode of the 200-ms mean within +-%g s, bins %g x the local range, "
              "accepted when >= %g of the window lies above and below it), else the 60-s moving mean"
              % (half_win_sec, bin_frac, side_frac))
    return z, method, float(np.mean(ok))


def _pair_areas(r1, a, b, s3win):
    """Inhale area I = sum of r > 0 from ``a`` to the first crossing from
    above to below zero flow; exhale area E = sum of -r (r < 0) from that
    crossing to ``b``; no crossing: I over [a, b], E = 0. Returns (I, E,
    crossing sample or NaN); ``r1`` is 1-padded, samples 1-based."""
    w = r1[a:b + 1]
    if s3win:                                           # the whole-window rule (measurement only)
        return np.sum(np.maximum(w, 0)), np.sum(np.maximum(-w, 0)), np.nan
    hit = np.nonzero((w[:-1] > 0) & (w[1:] <= 0))[0]
    if hit.size == 0:
        return np.sum(np.maximum(w, 0)), 0.0, np.nan
    i = int(hit[0]) + 1                                 # MATLAB's 1-based position of the crossing
    return np.sum(np.maximum(w[:i], 0)), np.sum(np.maximum(-w[i:], 0)), float(a + i)


def merge_incomplete_exhales(resp, fs, onsets, peaks, troughs, ex_frac=0.20, opts=None):
    """An inhale is not a new breath unless there has been an exhale since
    the last one.

    Port of ``mergeIncompleteExhales_zlp.m`` (rule text in
    ``docs/zlp_segmentation.md``): breaths in which a second inhale follows
    the first without an exhale in between (a double inhale, a staircase
    inhale, a chain of pulses at slow paced breathing, a small bump in a
    pause) are joined into one breath. An inhale joins the previous breath
    when the exhale since the previous inhale is under ``ex_frac`` (20 %) of
    that inhale, both measured as signal areas of the respiration trace
    above / below zero flow; pause blips and lead bumps join the breath
    before; the joined breath keeps its first onset unless that onset is the
    sensor's rebound off an exhale trough.

    Parameters
    ----------
    resp : respiration trace, raw units, inhale positive (NaN tolerated).
    fs : sampling rate (Hz).
    onsets : ``m`` ascending 0-based inhale-onset samples. Breath ``k`` is
        ``[onsets[k], onsets[k+1])``; the last onset only closes the last
        breath.
    peaks, troughs : ``m - 1`` samples (one per breath; ``m`` accepted, the
        last entry ignored).
    ex_frac : the exhale fraction below which an inhale joins the breath
        before.
    opts : dict of overrides of ``MERGE_DEFAULTS`` (snake_case; unknown
        names raise ``ValueError``).

    Returns
    -------
    ``(merged, info)``. ``merged`` - dict with one entry per merged breath:
    ``onsets, ends, peaks, troughs`` (0-based samples), ``kept_row`` (the
    input breath whose onset the merged breath keeps), ``n_pulses``,
    ``n_rows``, ``inhale_area``, ``exhale_area``, ``ex_in_ratio`` (areas of
    the whole merged breath, signal units x ms), ``orig_rows`` (list of
    arrays: the input breaths of each merged breath, ascending; first =
    ``kept_row``). ``info`` - the record of the rule: ``rule`` (text),
    ``version``, ``n_before``, ``n_after``, ``n_minor``, ``n_lead_bumps``,
    ``n_merged``, ``n_trough_heads`` and, per input breath, ``is_minor``,
    ``is_lead_bump``, ``pair_inhale_area``, ``pair_exhale_area``,
    ``pair_ex_in_ratio``, ``pair_cross_sec``, ``trough_left``,
    ``pause_left``, ``is_trough_head``, ``is_tail``, ``removal_ratio``,
    ``engine_onset_sec``, ... (all row indices 0-based).
    """
    o = _check_options(opts, _MERGE_DEFAULTS, "merge_incomplete_exhales")
    if ex_frac is None:
        ex_frac = 0.20
    x = _fillmissing_linear(resp)
    N = x.size
    on_in = np.array([_mround(v) for v in np.asarray(onsets, dtype=float).ravel()]) + 1   # 1-based
    m = on_in.size
    if m < 2:
        raise ValueError("at least two onsets (one breath) are needed")
    nB = m - 1
    pk_in = np.array([_mround(v) for v in np.asarray(peaks, dtype=float).ravel()]) + 1
    tr_in = np.array([_mround(v) for v in np.asarray(troughs, dtype=float).ravel()]) + 1
    if pk_in.size == m:
        pk_in = pk_in[:nB]
    if tr_in.size == m:
        tr_in = tr_in[:nB]
    if pk_in.size != nB or tr_in.size != nB:
        raise ValueError(f"peaks and troughs need one entry per breath ({nB}) or per onset ({m})")
    if not (np.all(np.diff(on_in) > 0) and on_in[0] >= 1 and on_in[-1] <= N):
        raise ValueError("onsets must be strictly ascending samples inside the trace")
    if not (np.all((pk_in >= 1) & (pk_in <= N)) and np.all((tr_in >= 1) & (tr_in <= N))):
        raise ValueError("peaks and troughs must be samples inside the trace")

    resp_raw = x - _movmean(x, _mround(60 * fs))
    tr_y = resp_raw[tr_in - 1]
    merged, info = _merge_core(on_in[:nB], on_in[1:], pk_in, tr_in, tr_y, x, fs, ex_frac, o)
    return merged, info


def _merge_core(on_s, end_s, pk_s, tr_s, tr_y, x, fs, ex_frac, o):
    """The rule on the breath table (per-breath 1-based onset, end, peak and
    trough samples). Rows are numbered from 1 inside (``[0]`` unused) to
    follow the MATLAB transcription; outputs are 0-based."""
    dead_frac, minor_frac = o["dead_frac"], o["minor_frac"]
    lead_frac, rest_sec = o["lead_frac"], o["rest_sec"]
    nbhd = 10                                       # rows each side for the local references
    N = x.size
    n0 = on_s.size
    x1 = _pad1(x)

    def rows(a):                                    # 1-padded per-row array
        return np.concatenate(([0], np.asarray(a)))

    on = rows(np.minimum(np.maximum(on_s, 1), N)).astype(int)
    en = rows(np.minimum(np.maximum(end_s, 1), N)).astype(int)
    pk = rows(np.minimum(np.maximum(pk_s, 1), N)).astype(int)
    trs = rows(tr_s).astype(int)
    trY = rows(tr_y)
    hiEnd = np.maximum(on, en - 1)                  # last sample of each detection window

    # ---- zero-flow level; inhale / exhale area densities (raw, and beyond the noise band) ----
    z, zmethod, zfrac = _zero_flow_level(x, fs, o["zero_half_win_sec"], o["zero_bin_frac"],
                                         o["zero_side_frac"], o["zero_max_mean"], o["zero_mass_frac"])
    z1 = _pad1(z)
    r = _movmean(x, max(1, _mround(0.2 * fs))) - z
    rp = _pad1(r)
    swing = np.full(n0 + 1, np.nan)
    for k in range(1, n0 + 1):
        w = rp[on[k]:hiEnd[k] + 1]
        swing[k] = w.max() - w.min()
    dRow = np.full(n0 + 1, np.nan)
    for k in range(1, n0 + 1):
        dRow[k] = dead_frac * _nanmedian(swing[max(1, k - nbhd):min(n0, k + nbhd) + 1])
    dS1 = np.zeros(N + 1)                           # noise band per sample (that of its detection row)
    for k in range(1, n0 + 1):
        dS1[on[k]:hiEnd[k] + 1] = dRow[k]
    dS1[1:on[1]] = dRow[1]
    dS1[hiEnd[n0] + 1:] = dRow[n0]
    dS = dS1[1:]
    zero = np.zeros(1)
    cI = np.concatenate((zero, np.cumsum(np.maximum(r, 0))))       # window [a, b] sum = c[b] - c[a - 1]
    cE = np.concatenate((zero, np.cumsum(np.maximum(-r, 0))))
    r1 = _movmean(x, max(1, _mround(o["blip_smooth_sec"] * fs))) - z    # steps 1-2: 1-s mean
    r1p = _pad1(r1)
    cIb = np.concatenate((zero, np.cumsum(np.maximum(r1 - dS, 0))))
    cIb02 = np.concatenate((zero, np.cumsum(np.maximum(r - dS, 0))))
    cEb = np.concatenate((zero, np.cumsum(np.maximum(-r1 - dS, 0))))
    cEb02 = np.concatenate((zero, np.cumsum(np.maximum(-r - dS, 0))))

    def areaI(a, b):
        return cI[b] - cI[a - 1]

    def areaE(a, b):
        return cE[b] - cE[a - 1]

    # ---- step 1: pause blips (no inhale, no exhale) join the breath before ----
    Ik = np.full(n0 + 1, np.nan)
    Ek = np.full(n0 + 1, np.nan)
    for k in range(1, n0 + 1):
        a, b = on[k], hiEnd[k]
        if b - a >= 4 * fs:
            Ik[k] = cIb[b] - cIb[a - 1]
            Ek[k] = cEb[b] - cEb[a - 1]
        else:
            Ik[k] = cIb02[b] - cIb02[a - 1]
            Ek[k] = cEb02[b] - cEb02[a - 1]
    isMinor = np.zeros(n0 + 1, dtype=bool)
    for k in range(2, n0 + 1):
        nb = list(range(max(1, k - nbhd), k)) + list(range(k + 1, min(n0, k + nbhd) + 1))
        ref = _nanmedian(Ik[nb])
        refE = _nanmedian(Ek[nb])
        isMinor[k] = ref > 0 and Ik[k] < minor_frac * ref and Ek[k] < minor_frac * refE
    # ---- step 2: a small pulse that only leads into a rest before the next inhale is a pause bump ----
    pk1 = np.full(n0 + 1, np.nan)
    for k in range(1, n0 + 1):
        pk1[k] = rp[on[k]:hiEnd[k] + 1].max()
    isLead = np.zeros(n0 + 1, dtype=bool)
    inh0 = [0] + [k for k in range(1, n0 + 1) if not isMinor[k]]
    nI0 = len(inh0) - 1
    for j in range(2, nI0):
        k, kn = inh0[j], inh0[j + 1]
        a = on[k]
        b = max(a, on[kn] - 1)
        nxt = pk1[kn]
        mm = j + 1                                  # highest pulse of the run the next inhale starts
        while mm < nI0:
            a2 = on[inh0[mm]]
            b2 = max(a2, on[inh0[mm + 1]] - 1)
            if areaE(a2, b2) >= ex_frac * areaI(a2, b2):
                break
            mm += 1
            nxt = max(nxt, pk1[inh0[mm]])
        if areaE(a, b) >= ex_frac * areaI(a, b) or pk1[k] >= lead_frac * nxt:
            continue
        if np.sum(np.abs(r1p[a:b + 1]) <= 2 * dS1[a:b + 1]) / fs >= rest_sec:   # time at rest in its window
            isLead[k] = True
            isMinor[k] = True
    inh = [0] + [k for k in range(1, n0 + 1) if not isMinor[k]]   # inhale rows (row 1 always)
    nI = len(inh) - 1

    # ---- step 3: exhale since the previous inhale vs that inhale ----
    s3win = o["step3_mode"] == "window"
    keepFirstOnly = o["keep_rule"] == "first"
    pairR = np.full(n0 + 1, np.nan)
    pairI = np.full(n0 + 1, np.nan)
    pairE = np.full(n0 + 1, np.nan)
    pairX = np.full(n0 + 1, np.nan)
    joinI = np.zeros(nI + 1, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(2, nI + 1):
            a = on[inh[j - 1]]
            b = max(a, on[inh[j]] - 1)
            I, E, cx = _pair_areas(rp, a, b, s3win)
            kk = inh[j - 1]
            pairI[kk], pairE[kk], pairX[kk] = I, E, cx
            pairR[kk] = np.float64(E) / np.float64(I)
            joinI[j] = E < ex_frac * I
        if nI >= 1:
            a = on[inh[nI]]
            b = hiEnd[n0]
            I, E, cx = _pair_areas(rp, a, b, s3win)
            kk = inh[nI]
            pairI[kk], pairE[kk], pairX[kk] = I, E, cx
            pairR[kk] = np.float64(E) / np.float64(I)
    # which onset the joined breath keeps (keep rule: see the MATLAB header / docs)
    troughWin = max(1, _mround(o["trough_win_sec"] * fs))
    troughFrac = o["trough_frac"]
    pauseWin = max(1, _mround(o["pause_win_sec"] * fs))
    pauseFrac = o["pause_frac"]
    pauseLvl = o["pause_level_frac"]
    pauseGap = max(0, _mround(o["pause_gap_sec"] * fs))
    swLoc = np.full(n0 + 1, np.nan)
    for k in range(1, n0 + 1):
        swLoc[k] = _nanmedian(swing[max(1, k - nbhd):min(n0, k + nbhd) + 1])
    troughLeft = np.zeros(n0 + 1, dtype=bool)
    pauseLeft = np.zeros(n0 + 1, dtype=bool)
    for k in range(1, n0 + 1):
        ok = on[k]
        wt = rp[max(1, ok - troughWin):ok + 1]
        troughLeft[k] = wt.min() <= -troughFrac * swLoc[k]
        wp = rp[max(1, ok - pauseGap - pauseWin):max(1, ok - pauseGap) + 1]
        pauseLeft[k] = (wp.max() - wp.min()) <= pauseFrac * swLoc[k] and abs(np.mean(wp)) <= pauseLvl * swLoc[k]
    grp = np.zeros(n0 + 1, dtype=int)
    head = np.zeros(n0 + 2, dtype=int)
    g = 0
    isTail = np.zeros(n0 + 1, dtype=bool)
    isTroughHead = np.zeros(n0 + 1, dtype=bool)
    removalRatio = np.full(n0 + 1, np.nan)
    jOf = np.zeros(n0 + 1, dtype=int)
    for j in range(1, nI + 1):
        jOf[inh[j]] = j
    for k in range(1, n0 + 1):
        if isMinor[k]:
            grp[k] = max(g, 1)
            continue
        j = jOf[k]
        if j == 1 or not joinI[j]:
            g += 1
            grp[k] = g
            head[g] = k
            continue
        h = head[g]
        rr = pairR[inh[j - 1]]
        if (not keepFirstOnly) and g > 1 and troughLeft[h] and pauseLeft[k]:
            mv = np.nonzero(grp == g)[0]            # the head and the rows it holds become the end of the breath before
            grp[mv] = g - 1
            isTail[mv] = True
            isTroughHead[h] = True
            removalRatio[h] = rr
            grp[k] = g
            head[g] = k
        else:
            grp[k] = g
            removalRatio[k] = rr                    # keep-first: the joining inhale is removed

    # ---- build the merged rows ----
    nK = int(grp.max())
    out_on, out_end, out_pk, out_tr = [], [], [], []
    keptRow, nP, nR, origRows = [], [], [], []
    inA, exA = [], []
    for q in range(1, nK + 1):
        mem = np.nonzero(grp == q)[0]
        kr, last = int(mem[0]), int(mem[-1])
        keptRow.append(kr - 1)
        nP.append(int(np.sum(~isMinor[mem] & ~isTail[mem])))
        nR.append(mem.size)
        origRows.append(mem - 1)
        pkSample, trSample = pk[kr], trs[kr]
        if mem.size > 1:
            pc = mem[~isTail[mem]]                  # a trough rebound taken over as the end of this breath is not its inhale peak
            ip = int(np.argmax(x1[pk[pc]] - z1[pk[pc]]))
            pkSample = pk[pc[ip]]
            it = int(np.argmin(trY[mem]))
            trSample = trs[mem[it]]
        out_on.append(on[kr] - 1)
        out_end.append(en[last] - 1)
        out_pk.append(pkSample - 1)
        out_tr.append(trSample - 1)
        a, b = on[kr], hiEnd[last]
        inA.append(areaI(a, b) / fs * 1000)
        exA.append(areaE(a, b) / fs * 1000)
    inA = np.array(inA)
    exA = np.array(exA)
    with np.errstate(divide="ignore", invalid="ignore"):
        exIn = exA / inA
    merged = dict(onsets=np.array(out_on, dtype=int), ends=np.array(out_end, dtype=int),
                  peaks=np.array(out_pk, dtype=int), troughs=np.array(out_tr, dtype=int),
                  kept_row=np.array(keptRow, dtype=int), n_pulses=np.array(nP, dtype=int),
                  n_rows=np.array(nR, dtype=int), inhale_area=inA, exhale_area=exA,
                  ex_in_ratio=exIn, orig_rows=origRows)
    nMinor = int(isMinor[1:].sum())
    rule = ("(2026-10-29a) an inhale is not a new breath when the exhale since the previous inhale is < %g x "
            "that inhale (inhale = area above zero flow from the previous onset to its first crossing below zero flow; "
            "exhale = area below zero flow from that crossing to the onset, 0 without a crossing); the joined breath "
            "keeps the earlier onset unless it has an exhale trough just before it and the joining inhale a flat pause "
            "just before it (then the earlier onset ends the breath before and the joining inhale starts the breath); "
            "engine onsets whose window holds neither an inhale nor an exhale (< %g x the local median beyond a +-%g x "
            "swing noise band), and small pulses that only lead into a >= %g-s rest before the next inhale, join the "
            "breath before" % (ex_frac, minor_frac, dead_frac, rest_sec))
    zA = z[on[1:] - 1]
    info = dict(
        rule=rule, version=MERGE_VERSION, ex_frac=ex_frac, dead_frac=dead_frac, minor_frac=minor_frac,
        zero_flow=zmethod, zero_flow_frac_dwell=zfrac, n_before=n0, n_after=nK, n_minor=nMinor,
        n_lead_bumps=int(isLead[1:].sum()), n_merged=n0 - nK - nMinor,
        n_pulses=merged["n_pulses"], n_rows=merged["n_rows"], inhale_area=inA, exhale_area=exA,
        ex_in_ratio=exIn, orig_rows=origRows, is_minor=isMinor[1:].copy(), is_lead_bump=isLead[1:].copy(),
        pulse_inhale_area=Ik[1:] / fs * 1000, pair_ex_in_ratio=pairR[1:],
        engine_onset_sec=(on[1:] - 1) / fs, zero_flow_at_onsets=zA, dead_band=dRow[1:],
        pair_inhale_area=pairI[1:] / fs * 1000, pair_exhale_area=pairE[1:] / fs * 1000,
        pair_cross_sec=(pairX[1:] - 1) / fs, trough_left=troughLeft[1:], pause_left=pauseLeft[1:],
        is_trough_head=isTroughHead[1:], is_tail=isTail[1:], removal_ratio=removalRatio[1:],
        n_trough_heads=int(isTroughHead[1:].sum()),
        keep_opts=dict(trough_win_sec=troughWin / fs, trough_frac=troughFrac, pause_win_sec=pauseWin / fs,
                       pause_frac=pauseFrac, pause_level_frac=pauseLvl, pause_gap_sec=pauseGap / fs,
                       step3_mode=o["step3_mode"], keep_rule=o["keep_rule"]))
    return merged, info


# the segment_breaths keyword of the same name shadows the function there
_merge_stage = merge_incomplete_exhales


def _apply_onset_refinement(rsp_stored, fs, on_in, pk_in, tr_in, opts):
    """Call ``refine_inhale_onsets`` on the detection's landmarks and CHECK
    what comes back (no repair: a violation raises ``ValueError``): onsets
    finite, inside the trace, strictly ascending; every onset has a peak
    after it; two onsets never share a peak. Returns ``(onsets, record)``;
    the record holds the detection's landmarks (``onsets_in`` ...), what went
    into the pairing (``onsets_out`` ..., peaks and troughs unchanged), the
    counts, ``rule`` / ``shift_ms`` aligned with ``onsets_out`` and the
    refinement's own info as ``refine_fn``."""
    on_in = np.asarray(on_in, dtype=float).ravel()
    pk_in = np.asarray(pk_in, dtype=float).ravel()
    tr_in = np.asarray(tr_in, dtype=float).ravel()
    on, rinfo = refine_inhale_onsets(rsp_stored, fs, on_in, pk_in, tr_in, opts)
    N = np.asarray(rsp_stored).size
    pk = np.sort(pk_in).astype(int)
    tr = np.sort(tr_in).astype(int)

    def bad(msg):
        raise ValueError(f"onset refinement result rejected: {msg}")

    if np.any(on < 0) or np.any(on > N - 1):
        bad("onsets must be finite and inside the trace")
    if on.size != on_in.size:
        bad(f"{on.size} onsets came back for {on_in.size}")
    if np.any(np.diff(on) <= 0):
        bad("onsets must be strictly ascending after rounding (duplicates)")
    n_on = on.size
    pair_pk = np.full(n_on, -1)
    for k in range(n_on):
        q = pk[pk > on[k]]
        if q.size == 0:
            bad(f"onset {k} (sample {on[k]}) has no peak after it")
        pair_pk[k] = q[0]
        if k < n_on - 1 and pair_pk[k] >= on[k + 1]:
            bad(f"onsets {k} and {k + 1} (samples {on[k]}, {on[k + 1]}) share the peak at sample {pair_pk[k]}")
    first_in = []
    for k in range(on_in.size):
        q = pk_in[pk_in > on_in[k]]
        if q.size:
            first_in.append(q[0])
    return on, dict(
        onsets_in=on_in.astype(int), peaks_in=pk_in.astype(int), troughs_in=tr_in.astype(int),
        onsets_out=on, peaks_out=pk, troughs_out=tr, n_on_in=on_in.size, n_on_out=n_on,
        n_peaks_in=pk_in.size, n_peaks_out=pk.size,
        n_orphan_peaks_in=pk_in.size - len(set(first_in)),
        n_orphan_peaks_out=pk.size - len(set(pair_pk.tolist())),
        rule=list(rinfo["rule"]), shift_ms=np.asarray(rinfo["shift_ms"], dtype=float), refine_fn=rinfo)


# --------------------------------------------------------------------------
# Driver + raw-unit measures
# --------------------------------------------------------------------------

def respiratory_volume_integrals(trace, fs, onsets, offsets):
    """breathmetrics' findRespiratoryVolumes integral on an arbitrary trace:
    sum(abs(trace[onset..offset]))/fs*1000 per breath, NaN where the offset
    is undefined, indices clamped to the trace bounds. 0-based inclusive."""
    trace = np.asarray(trace, dtype=float).ravel()
    n = trace.size
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    out = np.full(onsets.size, np.nan)
    for k in range(onsets.size):
        if k < offsets.size and np.isfinite(onsets[k]) and np.isfinite(offsets[k]):
            a = max(0, int(round(onsets[k])))
            b = min(n - 1, int(round(offsets[k])))
            if b >= a:
                out[k] = np.sum(np.abs(trace[a:b + 1])) / fs * 1000
    return out


def segment_breaths(resp, fs, blank_below_frac=None, sigh_span=None,
                    floor_frac=0.05, refine_onsets=True, refine_opts=None,
                    merge_incomplete_exhales=False, merge_ex_frac=0.20,
                    merge_opts=None):
    """Segment a respiration trace with the locked ZLP configuration and
    return per-breath measures computable from the landmarks alone.

    Two stages follow the detection, as in ``zlpEstimateAllFeatures.m``:
    the onset refinement (``refine_onsets``, ON by default: every inhale
    onset is re-placed on the stored trace, see ``refine_inhale_onsets``;
    ``refine_onsets=False`` gives the onsets of the detection as they were)
    and the incomplete-exhale merge (``merge_incomplete_exhales``, OFF by
    default: an inhale with no exhale since the previous one joins that
    breath, see ``merge_incomplete_exhales``; the per-breath measures below
    are then computed for each whole merged breath).

    Returns a dict of equal-length numpy arrays (one entry per breath; a
    breath spans inhale onset -> next inhale onset, so the final inhale is
    dropped, matching the MATLAB contracts):

    ``onset_idx, peak_idx, trough_idx, next_onset_idx`` - 0-based sample
        indices (trough = the exhale trough inside the breath);
    ``onset_t, peak_t, trough_t, next_onset_t`` - seconds;
    ``duration_s`` (onset -> next onset), ``time_to_peak_s``;
    ``onset_y, peak_y, trough_y, next_onset_y, amplitude`` - raw-unit
        amplitudes on the baselined trace (60-s moving mean removed);
        ``amplitude`` = peak_y - (onset_y + next_onset_y)/2;
    ``volume_raw_total`` - the breathmetrics volume integral over the whole
        breath (onset -> next onset) on the raw-unit baselined trace;
    ``volume_raw_inhale_proxy`` / ``volume_raw_exhale_proxy`` - the same
        integral split AT THE PEAK. NOTE this split is a proxy: the MATLAB
        toolbox splits at pause-aware inhale/exhale offsets computed by the
        breathmetrics class, which is not ported here.

    Also returned under ``"_detection"``: the det trace and the full
    landmark arrays (onsets/peaks/troughs, after the refinement and the merge
    when they are on), for plotting and QC; under
    ``"_refine"`` the check record of the onset refinement (``None`` when it
    is off; fields as in ``zlpInfo.refine`` of the MATLAB driver, in
    snake_case, ``refine_fn`` = the info of ``refine_inhale_onsets``); and
    under ``"_merge"`` ``None`` or ``{"merged", "info", "n_trough_guard"}``
    (the outputs of ``merge_incomplete_exhales`` on the paired breaths, and
    the number of merged breaths whose lowest member trough did not lie
    after the merged peak and was replaced by the lowest member trough that
    does).
    """
    resp_filled = _fillmissing_linear(resp)
    det, pk0, tr0, _ = prep_breath_trace(resp_filled, fs, "conservative",
                                         blank_below_frac, sigh_span,
                                         floor_frac)
    on, pk_p, tr_p = find_inhale_onsets(det, fs, pk0, tr0, "kneeBacktrack",
                                        0.4, 1.25, 0.50, 0.10)
    refine_rec = None
    if refine_onsets:
        # re-place the onsets on the stored trace (peaks / troughs unchanged)
        on, refine_rec = _apply_onset_refinement(resp_filled, fs, on, pk_p,
                                                 tr_p, refine_opts)
        on = np.sort(on)
    if on.size < 3:
        raise ValueError(f"only {on.size} inhale onsets detected - not a "
                         "usable respiration trace")

    # per-inhale landmark pairing (breathmetrics convention): peak(k) = first
    # peak after onset k, trough(k) = first trough after that peak; trailing
    # inhales with no following trough are dropped
    n0 = on.size
    pk_a = np.full(n0, -1, dtype=int)
    tr_a = np.full(n0, -1, dtype=int)
    for k in range(n0):
        p = pk_p[pk_p > on[k]]
        if p.size == 0:
            continue
        pk_a[k] = p[0]
        t = tr_p[tr_p > p[0]]
        if t.size:
            tr_a[k] = t[0]
    keep = pk_a >= 0
    on, pk_a, tr_a = on[keep], pk_a[keep], tr_a[keep]
    while on.size and tr_a[-1] < 0:
        on, pk_a, tr_a = on[:-1], pk_a[:-1], tr_a[:-1]
    if on.size < 3:
        raise ValueError(f"only {on.size} complete breaths after pairing")

    # optional: join inhales that have no exhale since the previous one.
    # Judged on breaths 0..n-2 (windows [on[k], on[k+1])); the last paired
    # inhale has no closing onset and is kept as it is. The merged landmarks
    # replace the originals, so every measure below describes a whole breath.
    merge_rec = None
    if merge_incomplete_exhales:
        n = on.size
        mg, mi = _merge_stage(
            resp_filled, fs, on, pk_a[:n - 1], tr_a[:n - 1], merge_ex_frac,
            merge_opts)
        raw_bc0 = resp_filled - _movmean(resp_filled, _mround(60 * fs))
        m_tr = mg["troughs"].copy()
        n_guard = 0
        for q in range(mg["onsets"].size):
            if m_tr[q] <= mg["peaks"][q]:
                cand = tr_a[mg["orig_rows"][q]]
                cand = cand[cand > mg["peaks"][q]]
                m_tr[q] = cand[int(np.argmin(raw_bc0[cand]))]
                n_guard += 1
        mg["troughs"] = m_tr
        on = np.concatenate((mg["onsets"], on[n - 1:]))
        pk_a = np.concatenate((mg["peaks"], pk_a[n - 1:]))
        tr_a = np.concatenate((m_tr, tr_a[n - 1:]))
        if not np.all((on[:-1] < pk_a[:-1]) & (pk_a[:-1] < tr_a[:-1])
                      & (tr_a[:-1] < on[1:])):
            raise ValueError("merged landmarks are not in onset < peak < "
                             "trough < next onset order")
        merge_rec = {"merged": mg, "info": mi, "n_trough_guard": n_guard}
        if on.size < 3:
            raise ValueError(f"only {on.size} breaths after the merge")

    # raw-unit baselined trace (same convention as the MATLAB bmObj columns)
    resp_raw = resp_filled - _movmean(resp_filled, _mround(60 * fs))

    nb = on.size - 1                     # breath ends at the next onset
    o0, o1 = on[:nb], on[1:nb + 1]
    pk_b, tr_b = pk_a[:nb], tr_a[:nb]
    vol_total = respiratory_volume_integrals(resp_raw, fs, o0, o1)
    vol_in = respiratory_volume_integrals(resp_raw, fs, o0, pk_b)
    vol_ex = respiratory_volume_integrals(resp_raw, fs, pk_b, o1)

    return {
        "onset_idx": o0, "peak_idx": pk_b, "trough_idx": tr_b,
        "next_onset_idx": o1,
        "onset_t": o0 / fs, "peak_t": pk_b / fs, "trough_t": tr_b / fs,
        "next_onset_t": o1 / fs,
        "duration_s": (o1 - o0) / fs,
        "time_to_peak_s": (pk_b - o0) / fs,
        "onset_y": resp_raw[o0], "peak_y": resp_raw[pk_b],
        "trough_y": resp_raw[tr_b], "next_onset_y": resp_raw[o1],
        "amplitude": resp_raw[pk_b] - (resp_raw[o0] + resp_raw[o1]) / 2,
        "volume_raw_total": vol_total,
        "volume_raw_inhale_proxy": vol_in,
        "volume_raw_exhale_proxy": vol_ex,
        "_detection": {"det": det, "onsets": on, "peaks": pk_a,
                       "troughs": tr_a, "fs": fs},
        "_refine": refine_rec,
        "_merge": merge_rec,
    }
