# ZLP breath segmentation — Python port

A self-contained Python port (numpy + scipy) of the ZLP segmentation engine that lives in
`breathmetrics_functions/` on the MATLAB side: the detection front end (`prepBreathTrace_zlp.m`,
`findInhaleOnsets_zlp.m`) and the two stages that follow it, the onset refinement
(`refineInhaleOnsets_zlp.m`) and the incomplete-exhale breath merge (`mergeIncompleteExhales_zlp.m`).
Full algorithm specification and motivation: [`../docs/zlp_segmentation.md`](../docs/zlp_segmentation.md).

```python
from zlp_breathmetrics import segment_breaths

br = segment_breaths(resp, fs)              # detection + onset refinement (the MATLAB driver's default)
br["onset_t"], br["peak_t"], br["duration_s"], br["volume_raw_total"], ...

br = segment_breaths(resp, fs, merge_incomplete_exhales=True)   # also join inhales with no exhale between
br["_merge"]["info"]["n_merged"]            # the records of the two stages: br["_refine"], br["_merge"]
```

Lower-level access (mirrors the MATLAB signatures; 0-based samples in and out):

```python
from zlp_breathmetrics import (prep_breath_trace, find_inhale_onsets,
                               refine_inhale_onsets, merge_incomplete_exhales)

det, pk0, tr0, info = prep_breath_trace(resp, fs, "conservative")
onsets, peaks, troughs = find_inhale_onsets(det, fs, pk0, tr0,
                                            "kneeBacktrack", 0.4, 1.25, 0.50, 0.10)

# re-place each onset on the stored trace (peaks / troughs / breath count unchanged)
onsets2, rinfo = refine_inhale_onsets(resp, fs, onsets, peaks, troughs)
rinfo["rule"], rinfo["shift_ms"]            # '' | 'foot' | 'far' | 'level' (+ '+clamp'), and the shift in ms

# join breaths with no exhale between them: one peak and one trough per breath (m - 1 breaths for m onsets)
merged, minfo = merge_incomplete_exhales(resp, fs, onsets2, peaks_per_breath, troughs_per_breath)
merged["onsets"], merged["orig_rows"], minfo["n_merged"], minfo["n_trough_heads"]
```

Option overrides are dicts with the snake_case names of the MATLAB options
(`refine_inhale_onsets(..., opts={"level_guard": "rise"})`, `merge_incomplete_exhales(..., ex_frac=0.5,
opts={"keep_rule": "first"})`); an unknown name raises `ValueError`. As in `zlpEstimateAllFeatures.m`,
`segment_breaths` refines the onsets by default (`refine_onsets=False` gives the onsets of the detection
as they were) and merges only on request; the merge is run on the paired breaths after the refinement and
each merged breath is measured as a whole.

## Scope

Ported: detection-trace conditioning, validity-ruled extrema (all three modes; `'conservative'` is the
locked choice), the eligibility rules and all three onset methods (`'kneeBacktrack'` locked), the
paced-sigh merge rule, the onset refinement (rules `foot`, `far`, `level`), the incomplete-exhale merge
(pause blips, lead bumps, the exhale test and the keep rule), and raw-unit amplitude/volume measures
(`segment_breaths`, `respiratory_volume_integrals`).

NOT ported: the breathmetrics feature computer. In MATLAB the landmarks are injected into the unmodified
`breathmetrics` class (`zlpEstimateAllFeatures.m`), which then computes exhale onsets, pauses, offsets,
normalized volumes and the shape/secondary features. Python has no breathmetrics class, so
`segment_breaths` reports only measures well defined from the landmarks alone — note in particular that
its `volume_raw_inhale_proxy` / `volume_raw_exhale_proxy` split the breath **at the peak**, not at the
toolbox's pause-aware offsets.

## Conventions

- All returned sample indices are **0-based** (the MATLAB originals are 1-based); times are
  `sample / fs`.
- The detection trace is windowed-normalized (30-s moving std); amplitude and volume outputs of
  `segment_breaths` are computed on the raw-unit trace (60-s moving-mean baseline removed) — units are
  sensor units, not liters. The refinement and the merge work on the stored (raw-unit) trace.
- MATLAB-semantics helpers are replicated exactly: `movmean`/`movstd` centered shrinking windows
  (including the even-window asymmetry; the running sums are taken on the de-meaned trace), directional
  `movmax`/`movmin`/trailing `movsum`, `fillmissing` linear+nearest, round-half-away-from-zero, `findpeaks`
  with MATLAB's filter order (prominence first, then greedy-by-height minimum distance) rather than
  scipy's, `prctile` (midpoint rule), the colon operator `a:d:b` (the histogram edges of the zero-flow
  search), `histcounts`, `movmedian`, `interp1(..., 'linear', 'extrap')` and `conv(..., 'same')`.
  `conv` needs care: the zero-flow level of the merge takes the first maximum of a 3-bin smoothed
  histogram of integer counts, so equal counts tie, and which of the tied bins MATLAB picks is decided
  by the order in which it accumulates the floating-point sums (a plain sum, and for more than 32 bins
  a fused multiply-add chain). `_conv3_same` reproduces that order as measured against MATLAB R2026a;
  with a plain `numpy.convolve` the zero-flow level (and so the merge areas) differs by up to 8e-4
  relative on the sample recording, although the landmarks are the same.

## Parity with MATLAB

`make_parity_reference.m` (run in MATLAB from the repository root) exports two traces — the repository's
`sample_data.mat` and the deterministic synthetic trace of `tests/zlpSyntheticTrace.m`, which contains
double inhales, trough rebounds, lead bumps and pause blips so that every merge step and every
refinement rule fires — together with the outputs of every stage: the detection landmarks, the
refinement (default options and three option variants), the landmarks the MATLAB driver injects with the
stages off / refinement on / refinement + merge, and the merge (default options and two option
variants), run on the driver's refined paired landmarks. `parity_check.py` runs the port on the same
data and compares: every landmark sample, rule label, kept row, origin row and boolean flag must be
identical, real arrays (shifts, confidences, levels, amplitudes, areas, ratios, zero-flow levels,
crossing times) equal to within 1e-9 relative (elementwise, with a floor of 1e-9 x the array's largest
magnitude).

Result (MATLAB R2026a vs Python 3.12 / numpy 2.5 / scipy 1.18):

```
                 sample_data.mat (660,001 samples @ 1 kHz)   synthetic trace (318,800 samples @ 500 Hz)
detection        det trace max|diff| 5.1e-13; every landmark  det trace max|diff| 4.9e-13; every landmark
                 exact (prep 132/132 + 132/132, onset 131/131, exact (151/151 + 151/151, 150/150,
                 peak 132/132, trough 132/132)                151/151, 151/151)
refinement       131 onsets, 130 moved (foot 130); every      150 onsets, 146 moved (foot 141, far 1,
                 sample and rule label identical;             level 4); every sample and rule label
                 real arrays max rel 9.5e-14                  identical; real arrays max rel 2.0e-11
merge            130 -> 130 breaths (nothing to join); all    149 -> 131 breaths (6 pause blips / lead
                 fields identical; real arrays max rel        bumps, 12 joined inhales, 5 trough heads);
                 8.2e-14                                      all fields identical; max rel 1.8e-12
driver           landmarks identical in all three             landmarks identical in all three
                 configurations                               configurations (incl. the 132-onset merge)
option variants  3 refinement + 2 merge variants identical    3 refinement + 2 merge variants identical
                 (max rel 9.5e-14)                            (max rel 2.0e-11)
PARITY: PASS
```

The sample recording has no double inhales, so the merge changes nothing there; the non-trivial merge
and the `far` / `level` refinement rules are exercised by the synthetic trace. Python's `prctile` and
`interp1` are not bit-identical to MATLAB's (last-digit differences, the maximum relative differences
above), and the middle element of an odd-length colon range can differ in the last bit; none of this
changes a decision on these data. The generated `parity_ref/` folder is git-ignored (regenerate it with
the two scripts above).

## Requirements

Python ≥ 3.9, numpy, scipy (`scipy.signal.find_peaks`; `scipy.io.loadmat` for the parity check).
