# ZLP breath segmentation for breathmetrics

An alternative breath-segmentation front end for the `breathmetrics` class,
contributed by the Zelano Lab preprocessing pipeline (ZLP), where it is the
production engine behind every breathmetrics-based output across seven
task pipelines (intracranial + scalp-EEG sessions; belt, pressure and nasal
cannula respiration sensors).

**Entry point:** `zlpEstimateAllFeatures(resp, srate, ...)` — a drop-in
alternative to `bm.estimateAllFeatures()`.

```matlab
[bm, rawVolumes] = zlpEstimateAllFeatures(resp, srate);           % defaults
bm = zlpEstimateAllFeatures(resp, srate, 'sighSpan', [s0 s1]);    % paced sighs
bm = zlpEstimateAllFeatures(resp, srate, 'refineOnsets', false);  % onsets as before rev14
[bm, ~, zi] = zlpEstimateAllFeatures(resp, srate, ...
    'mergeIncompleteExhales', true);                              % one breath per exhale
```

Since rev14 the inhale onsets are re-placed on the stored trace after the
detection (section 3b, on by default), and an optional merge joins inhales
that have no exhale between them (section 3c, off by default).

---

## 1. Why

Stock breathmetrics finds inhale onsets as the last sample below a global
pause-band threshold in each trough→peak window, with extrema from
global-criterion peak detection. On amplitude-non-stationary recordings —
drifting baselines, breathing depth that changes several-fold between task
blocks, paced slow breathing, acquisition high-pass filters — that places
onsets on exhale troughs, on inhale peaks, or mid-rise, and either splits
single breaths or misses quiet ones depending on where the global criteria
are set.

The ZLP engine replaces breathmetrics' **extrema detection** and
**inhale-onset placement** with a purpose-built two-stage algorithm
(validity-ruled extrema + a slope-walk "kneeBacktrack" onset), while keeping
breathmetrics as the **feature computer**: the landmarks are injected into an
unmodified `breathmetrics` object and every downstream feature (exhale
onsets, pauses, offsets, durations, volumes, secondary statistics) is
recomputed by the class' own sanctioned post-manual-adjustment path
(`manualAdjustPostProcess`), so the whole feature set stays toolbox-native
and toolbox-versioned.

The algorithm was designed through roughly twenty live-reviewed diagnostic
generations plus several single-breath forensic drill-downs on the Zelano
Lab's corpus (≈ 80 task-sessions across seven paradigms, re-reviewed at each
generation), and each rule below exists because a reviewed failure case
demanded it. The configuration shipped here is the locked production version
(rev13, 2026-09), extended by the onset refinement (rev14, section 3b) and the
optional incomplete-exhale merge (section 3c).

---

## 2. What replaces what

| breathmetrics step | ZLP replacement |
|---|---|
| `findExtrema` / `findRespiratoryExtrema` | `prepBreathTrace_zlp` (stages 0–1) |
| inhale-onset placement in `findOnsetsAndPauses` | `findInhaleOnsets_zlp` (stages 2–3) |
| (rev14) onset position on the stored trace | `refineInhaleOnsets_zlp` (stage 4, on by default) |
| (optional) joining inhales that have no exhale between them | `mergeIncompleteExhales_zlp` (off by default) |
| everything downstream (exhale onsets, pauses, offsets, durations, volumes, secondary) | **unchanged breathmetrics**, recomputed from the injected landmarks |

Integration sequence inside `zlpEstimateAllFeatures` (usable as a recipe for
injecting any external segmentation):

```matlab
[det, pk0, tr0] = prepBreathTrace_zlp(resp, fs, 'conservative', blankBelowFrac, sighSpan, floorFrac);
[on, pkP, trP]  = findInhaleOnsets_zlp(det, fs, pk0, tr0, 'kneeBacktrack', 0.4, 1.25, 0.50, 0.10);
[on, pkP, trP]  = refineInhaleOnsets_zlp(resp, det, fs, on, pkP, trP);  % rev14 (checked by the driver)
% pair per-inhale arrays: inhalePeaks(k) = first peak after onset k,
% exhaleTroughs(k) = first trough after that peak; trailing inhales with no
% following trough are dropped (the class' simplify convention);
% optional: mergeIncompleteExhales_zlp on breaths 1..n-1, merged landmarks replace the pairs

bm = breathmetrics(det, fs, 'humanAirflow');
bm.correctRespirationToBaseline('sliding', 0, 0);
bm.inhalePeaks   = pkA;                       % OUR extrema
bm.exhaleTroughs = trA;
bm.peakInspiratoryFlows  = respBC(pkA);       % respBC = bm.baselineCorrectedRespiration
bm.troughExpiratoryFlows = respBC(trA);
bm.findOnsetsAndPauses(0);    % class derives exhale onsets + pauses from OUR extrema
bm.inhaleOnsets = on;         % REPLACE inhale onsets with the ZLP ones
% drop any pause onset the new onsets contradict (see the function), then:
bm.inhaleTimeToPeak = (pkA - on) / fs;
bm.manualAdjustPostProcess(); % class recomputes offsets, durations, volumes, secondary
```

`manualAdjustPostProcess` is breathmetrics' own "call this after manually
changing phase onsets" hook — the same path the GUI uses — so nothing inside
the toolbox needed changing.

---

## 3. The algorithm (all constants)

Detection runs on a normalized copy; **the input signal is never modified**.
All thresholds are in normalized units unless marked.

**Stage 0 — detection trace** (`prepBreathTrace_zlp`, common to all modes):

1. Linear NaN fill (nearest at the edges).
2. 500 ms moving-average smoothing.
3. Amplitude normalization: divide by the 30-s moving standard deviation,
   floored at `floorFrac` (default 0.05) × its median. This is what lets one
   prominence criterion serve quiet and deep-breathing epochs alike.
4. Optional blanking: samples whose local scale is below
   `blankBelowFrac` × median are zeroed (hardware-attenuated stretches, e.g.
   a partially unplugged cannula).

**Stage 1 — extrema** (`'conservative'` mode, the locked choice):

5. Peaks: prominence ≥ 0.6, min separation 1.0 s (`findpeaks`); troughs
   identically on the inverted trace.
6. Peak validity: height ≥ 0.5 (floor), **and** some point ≥ 0.5 below the
   peak within the following 1 s (soft descent) — waived for peaks ≥ 1.0.
   Kills exhale-recovery crests before breathing pauses, which pass both
   prominence and the rise filter because each measures against the deep
   trough below them.
7. Strict trough→peak→trough alternation (the more extreme of a same-type
   run wins).
8. Rise filter: a trough→peak rise under 40% of half the local 30-s range is
   a bump, not a breath ⇒ peak deleted.
9. Paced-sigh span only (`sighSpan`): of two peaks within 5 s, keep the
   FIRST — the sigh's top-up peak follows the primary inhale, so the paced
   double inhale segments as one breath.

**Stage 2 — onset eligibility** (`findInhaleOnsets_zlp`; slope = derivative
smoothed 120 ms):

10. Rule 1 (not descending): `x(t) − x(t−0.33 s) > −0.2`.
11. Rule 2 (rising ahead): within `max(0.4 s, 25% of the pair duration)`
    ahead, the trace must exceed `x(t)` by > 0.4. (The window is pair-scaled
    because a fixed 0.4-s window demands a rise rate no slow breath can
    meet.)
12. Eligible = rule 1 AND rule 2.
13. Spurious-pair pruning: a trough→peak window with no eligible sample is a
    false split — trough and peak are deleted and the region absorbs into
    the neighboring breath. (This is why the function returns pruned
    extrema: always take `peaks`/`troughs` from its outputs.)

**Stage 3 — placement (`kneeBacktrack`), one onset per surviving pair:**

14. `dmax` = max slope over the whole window (sets the walk-stop scale);
    the anchor's slope scale = max slope over the window's SECOND HALF only
    (on two-phase breaths a violent first rise otherwise sets a bar the
    final rise cannot meet).
15. Anchor: last point in the second half with slope ≥ 70% of that scale AND
    eligible; fallback: last eligible sample anywhere.
16. Walk back freely until slope < `0.50·dmax` sustained ≥ 0.10 s
    (knife-edge single-sample dips cannot stop the walk).
17. Clean sweep (only if the walk ran to the floor, below trough + 10% of
    the swing — pause-free breaths where no knee exists): onset = last
    upward crossing of the trough/peak midpoint.
18. Late-landing extension (landing above trough + 35%): keep walking under
    slope < `0.05·dmax` sustained 0.15 s — accept only if that stop actually
    fired; on an edge run-out revert to the main walk's landing.
19. Rule-3 refinement: slope contrast over −0.25..+0.5 s must satisfy
    max > 1.25 × max(min, 0.1); a contrast-less landing moves to the nearest
    contrasted sample in-window.
20. Final eligibility snap: a non-eligible landing moves to the nearest
    eligible sample (either direction).
21. Hard floor (applied last so nothing above can undo it): an onset never
    sits in the first 20% of its trough→peak interval — a landing before the
    floor moves forward to the first eligible sample at or after the floor
    (never backward), or to the floor itself if there is none.

---

## 3b. Onset refinement (stage 4, rev14)

The detection of section 3 finds each onset on the 500-ms-smoothed, 30-s-normalized trace. On the **stored**
trace (raw units) that onset is often early — the trace is still flat and the rise starts 0.1–0.3 s later —,
sometimes sits on a small bump well before a quick rise to the peak, and sometimes on a small upward
inflection riding high on a larger inhale. `refineInhaleOnsets_zlp` looks at the stored trace itself (lightly
smoothed) and moves an onset only when the trace shows a definitively better one. It runs right after
`findInhaleOnsets_zlp` (`'refineOnsets'`, default `true`); peaks and troughs are not changed and no breath is
added or dropped.

Slope: the stored trace is smoothed by moving means of 40, 80, 160 and 300 ms (central-difference slope over
the same span); per breath the finest scale whose steepest slope in the search range exceeds 5 × the local slope
noise (robust SD within ±10 s) is used. Foot of a rise: in a range before the peak, the steepest rise (weighted
by `exp(-(peak - t)/2.5 s)`, so of two similar rises the one closer to the peak wins), then walked back to the
last sample where the slope has stayed below max(5 % of the steepest slope, 2 × noise) for 50 ms, and moved
forward by 0.2 × the slope scale (the moving mean smears the start of a rise back by about half its span).

Rules, in order (each later rule sees the result of the earlier ones):

1. **`foot`** — early by smoothing. Search range [onset − 200 ms, peak − 100 ms], never before the floor. The foot
   replaces the onset when it moved by ≥ 20 ms. A forward move may step over a flat stretch or a small bump,
   never over part of the rise itself: it is refused when the level at the foot is more than 0.15 × the breath
   amplitude above the level at the detected onset (or more than 0.05 × the amplitude while the stepped-over
   stretch is itself steep); the foot of an earlier rise in the range is then tried (at most 3 times).
2. **`far`** — a bump long before a quick rise. An onset (after rule 1) farther from its peak than
   max(0.8 s, 1.5 × the median onset-to-peak time of the 8 neighbours on each side) is searched again in the last
   1.5 × that median before the peak; the foot found there replaces it (forward moves only; a slow stretch may skip
   up to 0.40 × the amplitude of level).
3. **`level`** — an onset riding high on a larger inhale. The reference level is the median onset level (after
   rules 1–2) of the whole trace; the margin is half the width of the middle 30 % of the raw trace values (65th
   minus 35th percentile). An onset above reference + margin is searched again from the floor: the foot of the
   steepest rise; while that foot is still above the band, the foot of the earlier rise; a foot more than 1.0 ×
   the median amplitude below the reference is refused (an exhale dip, not this inhale's start). Accepted when the
   new onset is lower by ≥ 0.10 × the amplitude. Guards: an onset that follows a flat pause near the usual onset
   level is not moved, and a pause that precedes the detected onset is a floor; a new onset with a deep exhale
   trough within 1 s before it is refused (a rebound, not an inhale start).

Floor and alternation: every search starts no earlier than max(previous peak + 50 ms, the pair's trough), and a
final pass keeps every onset strictly between the previous peak and its own peak and strictly ascending. The
driver checks the result (`zlpEstimateAllFeatures:inconsistentRefine`): onsets finite and inside the trace,
strictly ascending after rounding, one per detected onset, every onset has a peak after it, no two onsets share a
peak. All constants are fields of the options struct (`'refineOpts'`, see the function header); the defaults are the
validated configuration. `zlpInfo.refine` holds the landmarks before / after, the rule that moved each onset
(`''`, `'foot'`, `'far'`, `'level'`, `+clamp`) and the shift in ms.

`'refineOnsets', false` gives exactly the landmarks of the earlier version of the driver; on `sample_data.mat` the
`breathmetrics` object is then identical, property by property (all 39), to the one the previous version builds.

On `sample_data.mat` (660 s at 1 kHz; 131 detected onsets, 130 breaths) the refinement moves 130 of 131 onsets
(rule `foot`, all later: median shift 263 ms, 90th percentile 511 ms, maximum 911 ms).

---

## 3c. Joining inhales that have no exhale between them (optional)

An inhale is not a new breath unless there has been an exhale since the last one. `mergeIncompleteExhales_zlp(resp,
srate, onsets, peaks, troughs [, exFrac [, opts]])` takes the landmarks of the breaths (breath *k* = `[onsets(k),
onsets(k+1))`, one peak and one trough each) and joins an inhale to the breath before it when the exhale since
the previous inhale is under `exFrac` (0.20) of that inhale — both as signal areas of the trace above / below zero
flow. In `zlpEstimateAllFeatures` it is the option `'mergeIncompleteExhales'` (default `false`), run on the paired
breaths 1…n−1 after the refinement. Three steps, in order:

1. **Pause blips.** An onset whose own window holds neither an inhale nor an exhale (inhale and exhale area each
   under 10 % of the median of the windows within ±10 rows, beyond a ±5 %-of-swing noise band) joins the breath
   before it, as part of its pause.
2. **Lead bumps.** A small pulse (peak under half of the highest pulse of the run the next inhale starts) with no
   exhale after it and ≥ 3 s near zero flow before the next inhale is the overshoot of the previous exhale's
   recovery, not the start of a breath: it joins the breath before, and the breath starts at the next inhale.
3. **Merge.** For each remaining inhale *k*, over [onset(*k*−1), onset(*k*)): I = the area above zero flow up to
   the first crossing from above to below zero flow (to onset(*k*) when there is none), E = the area below zero
   flow from that crossing to onset(*k*) (0 without a crossing). When E < 0.20 × I, inhale *k* joins the breath of
   inhale *k*−1; consecutive joins form one breath (double inhales, chains of pulses), and each inhale is judged on
   its own window, so a later complete exhale always ends the breath.

Keep rule: a joined breath keeps its first onset, **except** when that onset has an exhale trough immediately before
it and the joining inhale a flat pause immediately before it — the first onset is then the sensor's rebound off the
trough; it, and the rows up to the joining inhale, become the end of the breath before, and the joining inhale starts
the breath. The merged breath's end is that of its last member, its peak the highest of its own inhale pulses, its
trough the lowest member trough.

Zero flow is the higher of (a) the level the trace dwells at between breaths (its most frequent level within ±10 s,
accepted only when ≥ 15 % of the window sits within one bin of it and ≥ 10 % lies above and below it) and (b) the
60-s moving mean — with slow breathing a high-pass filter lifts each pause above the 60-s mean after a strong exhale,
and over a long pause that offset would add up to a whole inhale's area. Areas are signal areas of the 200-ms
smoothed trace, not litres; nasal pressure under-reads slow flow, so a very slow, shallow exhale after a fast inhale
can fall under 20 % and the next inhale then merges.

In the driver the merged landmarks are injected in place of the originals, so the class recomputes **every** feature
for each whole merged breath (the object holds one breath per merged breath; a merged breath's trough is the lowest
member trough that lies after its merged peak, counted in `zlpInfo.merge.nTroughGuard`). `zlpInfo.merge` holds the
function's outputs: `merged` (onsets, ends, peaks, troughs, kept row, number of inhale pulses, areas, original rows)
and `info` (the rule record: counts `nBefore`, `nAfter`, `nMinor` = blips + lead bumps, `nMerged`, `nTroughHeads`, and
per-breath arrays).

On `sample_data.mat` the merge joins nothing (130 → 130 breaths): the recording has no double inhales. On the
deterministic synthetic trace of `tests/zlpSyntheticTrace.m` (638 s at 500 Hz; regular breaths with double inhales,
trough rebounds, lead bumps, pause blips, quick inhales behind a slow bump, and two-stage inhales) the refinement
fires every rule (141 `foot`, 1 `far`, 4 `level`) and the merge takes 149 breaths to 131 (6 engine rows joined as pause
blips or lead bumps, 4 of them lead bumps; 12 inhales merged; 5 breaths starting at a later onset by the keep rule).

Both functions were checked against the Zelano Lab production implementations of the same rules: on
`sample_data.mat` and on the synthetic trace the refined onsets, rule labels and shifts, the merged landmarks and every
per-breath field of the merge record are identical.

---

## 4. Units — and the raw-volume twins

The `breathmetrics` object built by `zlpEstimateAllFeatures` holds the
**windowed-normalized detection trace**, so its flow and volume features are
in locally-rescaled units. That is deliberate (it is what makes detection
robust across epochs) and fine for timing analysis, but it makes amplitude
features **incomparable across epochs whose breathing depth differs**: the
30-s moving-std divisor tracks local depth, so a physically large breath
inside a deep-breathing stretch reads *smaller* than the same breath during
quiet breathing.

For interpretable amplitude analysis, request the second output:

```matlab
[bm, raw] = zlpEstimateAllFeatures(resp, srate);
raw.inhaleVolumesRaw   % toolbox volume integral, raw signal units
raw.exhaleVolumesRaw
```

These are computed by the toolbox's own `findRespiratoryVolumes` on the
raw-unit trace (60-s moving-mean baseline removed) between the **same**
onset/offset landmarks stored in `bm` — timing metrics are untouched; only
the integration trace differs. Units remain arbitrary sensor units (belt or
pressure voltage, not liters): within-recording and cross-condition
comparisons are meaningful; for between-subject comparisons normalize per
recording at analysis time.

Onsets/peaks/troughs are sample indices at `srate`; durations and
time-to-peak are seconds — identical to stock breathmetrics.

---

## 5. Caveats

- The onset thresholds are calibrated to the **normalized** trace of
  stage 0; feeding raw-unit signals into `findInhaleOnsets_zlp` directly
  will not work. Use the functions through `zlpEstimateAllFeatures` (or
  reproduce its conditioning).
- `findInhaleOnsets_zlp` can DELETE extrema (spurious-pair pruning) — always
  take `peaks`/`troughs` from its outputs, not from `prepBreathTrace_zlp`'s.
- Provide `sighSpan` for recordings containing a paced double-inhale (sigh)
  block; without it the double inhale is segmented as two breaths.
- Inhale-pause onsets come from the class' `findOnsetsAndPauses` run on the
  injected extrema; after the inhale onsets are replaced, any pause onset
  the new onsets contradict is set to NaN (no pause) rather than being
  re-estimated. If your analysis leans heavily on inhale-pause detection,
  inspect those estimates.
- Airflow data types only (`humanAirflow` / `rodentAirflow`) — the
  manual-adjust path exists only for them.
- `refineInhaleOnsets_zlp` works on the stored (raw-unit) trace and, unlike the
  detection, needs no normalization; its constants are in seconds, not samples.
- The merge changes the number of breaths in the `breathmetrics` object; code that
  matches breaths between two runs (with and without the merge, or before and after
  the refinement) must match by onset sample, not by index.
- Requirements: base MATLAB R2017b+ and the Signal Processing Toolbox
  (`findpeaks`, used by the `'conservative'` extrema mode; the `'pwl'` mode
  is toolbox-free). The merge also needs `prctile` (Statistics and Machine
  Learning Toolbox).

---

## 6. Provenance

Developed and locked in the Zelano Lab preprocessing repository
(`zelanoLabPreprocessing`, where the same two algorithm functions run in
production) over an August–September 2026 QC campaign: ~20 live-reviewed
diagnostic generations on ≈ 80 task-sessions spanning seven paradigms
(free breathing, paced slow breathing and sighs, audiobook listening,
emotional films, meditation, sleep), with per-breath forensic drill-downs
driving each rule. Every constant in §3 is the reviewed production value;
the code here is kept line-identical to the production copies so fixes can
flow both ways.
