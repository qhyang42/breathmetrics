# ZLP breath segmentation — Python port

A self-contained Python port (numpy + scipy) of the ZLP segmentation front
end that lives in `breathmetrics_functions/` on the MATLAB side
(`prepBreathTrace_zlp.m`, `findInhaleOnsets_zlp.m`). Full algorithm
specification and motivation: [`../docs/zlp_segmentation.md`](../docs/zlp_segmentation.md).

```python
from zlp_breathmetrics import segment_breaths

br = segment_breaths(resp, fs)              # locked production configuration
br["onset_t"], br["peak_t"], br["duration_s"], br["volume_raw_total"], ...
```

Lower-level access (mirrors the MATLAB signatures):

```python
from zlp_breathmetrics import prep_breath_trace, find_inhale_onsets

det, pk0, tr0, info = prep_breath_trace(resp, fs, "conservative")
onsets, peaks, troughs = find_inhale_onsets(det, fs, pk0, tr0,
                                            "kneeBacktrack", 0.4, 1.25, 0.50, 0.10)
```

## Scope

Ported: detection-trace conditioning, validity-ruled extrema (all three
modes; `'conservative'` is the locked choice), the eligibility rules and all
three onset methods (`'kneeBacktrack'` locked), the paced-sigh merge rule,
and raw-unit amplitude/volume measures (`segment_breaths`,
`respiratory_volume_integrals`).

NOT ported: the breathmetrics feature computer. In MATLAB the landmarks are
injected into the unmodified `breathmetrics` class
(`zlpEstimateAllFeatures.m`), which then computes exhale onsets, pauses,
offsets, normalized volumes and the shape/secondary features. Python has no
breathmetrics class, so `segment_breaths` reports only measures well defined
from the landmarks alone — note in particular that its
`volume_raw_inhale_proxy` / `volume_raw_exhale_proxy` split the breath **at
the peak**, not at the toolbox's pause-aware offsets.

## Conventions

- All returned sample indices are **0-based** (the MATLAB originals are
  1-based).
- The detection trace is windowed-normalized (30-s moving std); amplitude
  and volume outputs of `segment_breaths` are computed on the raw-unit trace
  (60-s moving-mean baseline removed) — units are sensor units, not liters.
- MATLAB-semantics helpers are replicated exactly: `movmean`/`movstd`
  centered shrinking windows (including the even-window asymmetry),
  directional `movmax`/`movmin`/trailing `movsum`, `fillmissing`
  linear+nearest, round-half-away-from-zero, and `findpeaks` with
  MATLAB's filter order (prominence first, then greedy-by-height minimum
  distance) rather than scipy's.

## Parity with MATLAB

`make_parity_reference.m` (run in MATLAB from the repository root) exports
the sample recording, the conditioned detection trace and every landmark
family; `parity_check.py` then runs the port on the same data and compares.

Result on the repository's `sample_data.mat` (660,001 samples @ 1 kHz,
MATLAB R2024-era vs Python 3.12 / numpy 2.5 / scipy 1.18):

```
det trace    max|diff| = 5.2e-13   (floating-point noise)
prepPeak     132/132 exact
prepTrough   132/132 exact
onset        131/131 exact
peak         132/132 exact
trough       132/132 exact
PARITY: PASS
```

Every landmark is sample-for-sample identical to the MATLAB implementation.
The generated `parity_ref/` folder is git-ignored (regenerate it with the
two scripts above).

## Requirements

Python ≥ 3.9, numpy, scipy.
