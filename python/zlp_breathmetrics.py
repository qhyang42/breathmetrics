"""ZLP breath segmentation - Python port of the MATLAB implementation.

Port of the custom segmentation front end contributed to breathmetrics
(``breathmetrics_functions/prepBreathTrace_zlp.m`` and
``breathmetrics_functions/findInhaleOnsets_zlp.m``), plus a convenience
driver. The algorithm specification lives in ``docs/zlp_segmentation.md``;
this module reproduces the MATLAB functions constant for constant, with
helper routines that replicate MATLAB's ``movmean``/``movstd`` (centered,
shrinking windows), directional ``movmax``/``movmin``/``movsum``,
``fillmissing`` and ``findpeaks`` (prominence first, then MATLAB's
greedy-by-height minimum-distance rule) semantics.

Scope
-----
Ported here: detection-trace conditioning, validity-ruled extrema, the
eligibility rules + kneeBacktrack onset placement (with the slopeGate and
changepoint comparison methods), and raw-unit amplitude/volume measures.
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
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cnt = hi - lo + 1
    return (cs[hi + 1] - cs[lo]) / cnt


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
                    floor_frac=0.05):
    """Segment a respiration trace with the locked ZLP configuration and
    return per-breath measures computable from the landmarks alone.

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
    landmark arrays (onsets/peaks/troughs), for plotting and QC.
    """
    resp_filled = _fillmissing_linear(resp)
    det, pk0, tr0, _ = prep_breath_trace(resp_filled, fs, "conservative",
                                         blank_below_frac, sigh_span,
                                         floor_frac)
    on, pk_p, tr_p = find_inhale_onsets(det, fs, pk0, tr0, "kneeBacktrack",
                                        0.4, 1.25, 0.50, 0.10)
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
    }
