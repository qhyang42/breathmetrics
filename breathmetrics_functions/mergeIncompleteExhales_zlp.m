function [merged, info] = mergeIncompleteExhales_zlp(resp, srate, onsets, peaks, troughs, exFrac, opts)
%MERGEINCOMPLETEEXHALES_ZLP  An inhale is not a new breath unless there has been an exhale since the last one.
%
%   [merged, info] = mergeIncompleteExhales_zlp(resp, srate, onsets, peaks, troughs [, exFrac [, opts]])
%
%   Post-processing for the landmarks of the ZLP segmentation: breaths in which a second inhale follows the first without an
%   exhale in between (a double inhale, a staircase inhale, a chain of pulses at slow paced breathing, a small bump in a pause)
%   are joined into one breath.  The rule: an inhale joins the previous breath when the exhale since the previous inhale is
%   under exFrac (20 %) of that inhale, both measured as signal areas of the respiration trace above / below zero flow.
%
%   INPUTS
%   resp     respiration trace, numeric vector, RAW units, inhale positive (NaN tolerated, filled by linear interpolation).
%   srate    sampling rate (Hz).
%   onsets   1 x m ascending inhale-onset samples.  Breath k (k = 1..m-1) is [onsets(k), onsets(k+1)): its end is the next
%            onset, so the last onset only closes the last breath.  This is the row set of the breath table of
%            segmentBreaths-style engines (one row per pair of consecutive onsets).
%   peaks    1 x (m-1) inhale-peak sample of each breath;  troughs  1 x (m-1) exhale-trough sample of each breath.  A 1 x m
%            array is accepted and its last entry ignored (zlpEstimateAllFeatures keeps a peak and trough for the last onset).
%   exFrac   (0.20) the exhale fraction below which an inhale joins the breath before.
%   opts     struct of optional settings (defaults in brackets), see STEPS.
%
%   OUTPUTS
%   merged   struct, one entry per MERGED breath (1 x nAfter row vectors, samples):
%              onsets    start of the merged breath = the onset of its kept row (see KEEP RULE)
%              ends      end of the merged breath = the end of its last member (the next breath's onset)
%              peaks     the highest inhale peak among the breath's own inhale pulses (rebound rows taken over as the
%                        end of the breath are not counted)
%              troughs   the lowest exhale trough among all its member rows
%              keptRow   index of the input breath whose onset the merged breath keeps
%              nPulses   inhale pulses in the breath (1 = not merged)
%              nRows     input breaths it holds (pause blips included)
%              inhaleArea, exhaleArea, exInRatio   areas of the whole merged breath (signal units x ms) and their ratio
%              origRows  cell: the input breaths of each merged breath, ascending (origRows{q}(1) == keptRow(q))
%   info     the record of the rule (below).
%
%   STEPS (in order; the values in brackets are the defaults of opts).
%   1  Pause blips.  An onset whose own window [onset(k), onset(k+1)) holds neither an inhale nor an exhale - inhale area I_k
%      AND exhale area E_k each below minorFrac (0.10) x the median of the windows within +-10 rows - is not a breath
%      (typically the onset where an exhale returns to rest, the pause after it sitting a little above zero flow); it joins the
%      breath before it, as part of its pause.  Here I_k / E_k count the trace beyond a noise band of +-deadFrac (0.05) x the
%      local breath swing (median max - min of the windows within +-10 rows) around zero flow, on the blipSmoothSec (1 s) mean
%      of the trace for windows >= 4 s (cardiac ripple on a pause averages out; over a 7-s pause a few units add up) and on the
%      200-ms mean for shorter ones (a 1-s mean would cancel a fast breath).
%   2  Lead bumps.  An inhale k with no exhale before the next inhale (E < exFrac x I over its window), a peak under leadFrac
%      (0.5) x the highest pulse of the run the next inhale starts (its pulses up to the next exhale), and >= restSec (3 s) near
%      zero flow (1-s mean within 2 noise bands) in its window before the next onset is the overshoot of the previous exhale's
%      recovery followed by a pause, not the start of the next breath: it joins the breath before, and the breath starts at the
%      next inhale.  Contiguous pulses (a double inhale) have no rest between them and are left to step 3.
%   3  Merge.  For every remaining inhale k (in order), over [onset(k-1), onset(k)) - since the LAST inhale, merged or not,
%      blips included: I = area above zero flow from onset(k-1) to the first crossing from above to below zero flow (or to
%      onset(k) when there is none), E = area below zero flow from that crossing to onset(k) (0 without a crossing: no exhale
%      phase).  When E < exFrac (0.20) x I, inhale k is not a new breath and joins the breath of inhale k-1.  Consecutive joins
%      form one breath (a double inhale, a chain of pulses); each inhale is judged on its own window, so a later complete
%      exhale always ends the breath.
%
%   KEEP RULE (which onset a joined breath keeps).  The joined breath keeps its first onset (keep-first), EXCEPT when that onset
%   has an exhale trough immediately before it (minimum of the 200-ms mean over the troughWinSec (1 s) before it is at least
%   troughFrac (0.25) x the local swing below zero flow) and the joining inhale has a flat pause immediately before it (over
%   the pauseWinSec (0.5 s) that end pauseGapSec (0.1 s) before it: range <= pauseFrac (0.15) x swing, mean within
%   pauseLevelFrac (0.25) x swing of zero flow): then the first onset is the sensor's rebound off the trough (the start of the
%   pause, not an inhale), it and the rows up to the joining inhale become the END of the breath before, and the joining inhale
%   starts the breath.  Not for the first breath of the recording.
%
%   ZERO FLOW.  The zero-flow level z(t) is the HIGHER of (a) the level the trace dwells at between breaths - its most frequent
%   level within +-zeroHalfWinSec (10 s), accepted only when >= zeroMassFrac (15 %) of the window sits within one bin of it and
%   >= zeroSideFrac (10 %) lies above and below it - and (b) the 60-s moving mean.  Why: with slow breathing the pauses are zero
%   flow, but a high-pass filter in the recording chain lifts each pause above the 60-s mean after a strong exhale, and over a
%   long pause that offset adds up to a whole inhale's area; without pauses the trace dwells on the slow part of the exhale
%   instead, BELOW zero flow, where the 60-s mean is the better estimate.  Not the detection's onset level (without a pause the
%   onsets sit low on the exhale recovery limb) and not a flatness test (cardiac ripple on the pauses defeats it).
%
%   AREAS are signal areas (sensor units x ms) of the 200-ms moving mean of the trace above / below zero flow, not litres.  Step 3
%   and the reported areas use no noise band: a slow exhale reads small on a pressure sensor (pressure ~ flow^2), and a band
%   would erase it.  Nasal pressure under-reads slow flow, so a very slow, shallow exhale after a fast inhale can fall under 20 %
%   of the inhale's signal area and the next inhale then merges.  They are not the volumes of the breathmetrics class, which
%   integrate |trace| over their own exhale window: a dip that never goes below zero flow (no exhale at all) still gets a
%   sizeable "exhale volume" there.
%
%   opts fields (all optional): minorFrac (0.10), deadFrac (0.05), blipSmoothSec (1) - step 1;  leadFrac (0.5), restSec (3) -
%   step 2;  troughWinSec (1), troughFrac (0.25), pauseWinSec (0.5), pauseFrac (0.15), pauseLevelFrac (0.25), pauseGapSec (0.1) -
%   the keep rule;  zeroHalfWinSec (10), zeroBinFrac (0.04), zeroSideFrac (0.10), zeroMassFrac (0.15), zeroMaxMean (true) - the
%   zero-flow level.  Measurement-only switches: step3Mode ('pairCross' = the rule above, 'window' = I and E as the areas above
%   / below zero flow over the whole window) and keepRule ('troughPause' = the rule above, 'first' = keep-first only).
%
%   info fields: rule (text), version, exFrac, deadFrac, minorFrac, zeroFlow (text), zeroFlowFracDwell (share of the recording
%   where the dwell level was accepted), nBefore, nAfter, nMinor (input breaths joined to the breath before as blips or lead
%   bumps), nLeadBumps (of them, step 2), nMerged (inhales merged into the breath before; nBefore - nAfter = nMinor + nMerged),
%   nTroughHeads (breaths that start at a later onset by the keep rule); per KEPT row nPulses, nRows, inhaleArea, exhaleArea,
%   exInRatio, origRows; per INPUT breath isMinor, isLeadBump, pulseInhaleArea (step 1's I_k), pairInhaleArea / pairExhaleArea /
%   pairExInRatio (step 3's I and E of [onset(k), next inhale onset) and their ratio; NaN for a blip), pairCrossSec (the
%   crossing sample / srate, NaN = none), troughLeft, pauseLeft, isTroughHead (a first onset removed by the keep rule), isTail
%   (rows that became the end of the breath before), removalRatio (E/I of the pair that removed the row), engineOnsetSec (the
%   input onsets / srate), zeroFlowAtOnsets, deadBand, keepOpts.
%
%   Needs prctile (Statistics and Machine Learning Toolbox) for the zero-flow level.

    if nargin < 6 || isempty(exFrac), exFrac = 0.20; end
    if nargin < 7 || isempty(opts), opts = struct(); end
    assert(isvector(resp) && isnumeric(resp), 'mergeIncompleteExhales_zlp:resp', 'resp must be a numeric vector');
    x = fillmissing(double(resp(:)'), 'linear', 'EndValues', 'nearest');
    N = numel(x);
    onsets = round(double(onsets(:)'));
    m = numel(onsets);
    assert(m >= 2, 'mergeIncompleteExhales_zlp:onsets', 'at least two onsets (one breath) are needed');
    nB = m - 1;
    peaks = round(double(peaks(:)')); troughs = round(double(troughs(:)'));
    if numel(peaks) == m, peaks = peaks(1:nB); end
    if numel(troughs) == m, troughs = troughs(1:nB); end
    assert(numel(peaks) == nB && numel(troughs) == nB, 'mergeIncompleteExhales_zlp:landmarks', ...
        'peaks and troughs need one entry per breath (%d) or per onset (%d)', nB, m);
    assert(all(diff(onsets) > 0) && onsets(1) >= 1 && onsets(end) <= N, 'mergeIncompleteExhales_zlp:onsets', ...
        'onsets must be strictly ascending samples inside the trace');
    assert(all(peaks >= 1 & peaks <= N & troughs >= 1 & troughs <= N), 'mergeIncompleteExhales_zlp:landmarks', ...
        'peaks and troughs must be samples inside the trace');

    % breath table in the layout the rule works on: columns 1-11 as the engine builds them (amplitudes from the trace with its
    % 60-s moving mean removed, times = sample / srate), column 14 the breath index
    respRaw = x - movmean(x, round(60 * srate));
    bmObj = zeros(nB, 14);
    bmObj(:, 1)  = respRaw(onsets(1:nB));
    bmObj(:, 2)  = onsets(1:nB) / srate;
    bmObj(:, 3)  = respRaw(peaks(1:nB));
    bmObj(:, 4)  = peaks(1:nB) / srate;
    bmObj(:, 5)  = respRaw(onsets(2:nB + 1));
    bmObj(:, 6)  = onsets(2:nB + 1) / srate;
    bmObj(:, 7)  = bmObj(:, 6) - bmObj(:, 2);
    bmObj(:, 8)  = bmObj(:, 3) - (bmObj(:, 1) + bmObj(:, 5)) / 2;
    bmObj(:, 9)  = peaks(1:nB);
    bmObj(:, 10) = respRaw(troughs(1:nB));
    bmObj(:, 11) = troughs(1:nB) / srate;
    bmObj(:, 14) = 1:nB;

    [bmOut, ~, info] = mergeCore(bmObj, struct(), x, srate, exFrac, opts);

    keptRow = cellfun(@(c) c(1), info.origRows(:))';
    merged = struct('onsets', round(bmOut(:, 2)' * srate), 'ends', round(bmOut(:, 6)' * srate), ...
        'peaks', bmOut(:, 9)', 'troughs', round(bmOut(:, 11)' * srate), 'keptRow', keptRow, ...
        'nPulses', info.nPulses(:)', 'nRows', info.nRows(:)', 'inhaleArea', info.inhaleArea(:)', ...
        'exhaleArea', info.exhaleArea(:)', 'exInRatio', info.exInRatio(:)', 'origRows', {info.origRows(:)'});
end

% ------------------------------------------------------------------------------------------------
% The rule itself, on the breath table built above (columns 1-11 and 14; bmFeatures is an empty struct here).
function [bmObj, bmFeatures, info] = mergeCore(bmObj, bmFeatures, rsp, fs, exFrac, opts)
    if nargin < 5 || isempty(exFrac), exFrac = 0.20; end
    if nargin < 6 || isempty(opts), opts = struct(); end
    deadFrac  = getOpt(opts, 'deadFrac', 0.05);
    minorFrac = getOpt(opts, 'minorFrac', 0.10);
    zHalf     = getOpt(opts, 'zeroHalfWinSec', 10);
    zBin      = getOpt(opts, 'zeroBinFrac', 0.04);
    zSide     = getOpt(opts, 'zeroSideFrac', 0.10);
    blipSm    = getOpt(opts, 'blipSmoothSec', 1);
    leadFrac  = getOpt(opts, 'leadFrac', 0.5);
    restSec   = getOpt(opts, 'restSec', 3);
    nbhd = 10;                                          % rows each side for the local references
    x = fillmissing(double(rsp(:)'), 'linear', 'EndValues', 'nearest');
    N = numel(x);
    n0 = size(bmObj, 1);
    on0Sec = bmObj(:, 2);
    on = min(max(round(bmObj(:, 2) * fs), 1), N);      % onset samples (the engine stores sample / fs)
    en = min(max(round(bmObj(:, 6) * fs), 1), N);      % breath end = next onset
    pk = min(max(round(bmObj(:, 4) * fs), 1), N);
    hiEnd = max(on, en - 1);                            % last sample of each engine window

    % ---- zero-flow level; inhale / exhale area densities (raw, and beyond the noise band) ----
    [z, zInfo] = zeroFlowLevel(x, fs, zHalf, zBin, zSide, getOpt(opts, 'zeroMaxMean', true), getOpt(opts, 'zeroMassFrac', 0.15));
    r = movmean(x, max(1, round(0.2 * fs))) - z;
    swing = nan(n0, 1);
    for k = 1:n0, w = r(on(k):hiEnd(k)); swing(k) = max(w) - min(w); end
    dRow = nan(n0, 1);
    for k = 1:n0, dRow(k) = deadFrac * median(swing(max(1, k - nbhd):min(n0, k + nbhd)), 'omitnan'); end
    dS = zeros(1, N);                                   % noise band per sample (that of its engine row)
    for k = 1:n0, dS(on(k):hiEnd(k)) = dRow(k); end
    if n0 >= 1, dS(1:on(1) - 1) = dRow(1); dS(hiEnd(n0) + 1:end) = dRow(n0); end
    cI = [0 cumsum(max(r, 0))];                         % window [a, b] sum = c(b + 1) - c(a)
    cE = [0 cumsum(max(-r, 0))];
    r1 = movmean(x, max(1, round(blipSm * fs))) - z;    % steps 1-2: 1-s mean (cardiac ripple on a pause averages out)
    cIb = [0 cumsum(max(r1 - dS, 0))];                  % beyond the noise band (step 1, windows >= 4 s)
    cIb02 = [0 cumsum(max(r - dS, 0))];                 % shorter windows: the 1-s mean would cancel a fast breath
    cEb = [0 cumsum(max(-r1 - dS, 0))]; cEb02 = [0 cumsum(max(-r - dS, 0))];
    areaI = @(a, b) cI(b + 1) - cI(a);
    areaE = @(a, b) cE(b + 1) - cE(a);

    % ---- step 1: pause blips (no inhale, no exhale) join the breath before ----
    Ik = nan(n0, 1); Ek = nan(n0, 1);
    for k = 1:n0
        a = on(k); b = hiEnd(k);
        if b - a >= 4 * fs, Ik(k) = cIb(b + 1) - cIb(a); Ek(k) = cEb(b + 1) - cEb(a);
        else, Ik(k) = cIb02(b + 1) - cIb02(a); Ek(k) = cEb02(b + 1) - cEb02(a); end
    end
    clear cIb02 cEb cEb02
    isMinor = false(n0, 1);
    for k = 2:n0
        nb = [max(1, k - nbhd):k - 1, k + 1:min(n0, k + nbhd)];
        ref = median(Ik(nb), 'omitnan'); refE = median(Ek(nb), 'omitnan');
        isMinor(k) = ref > 0 && Ik(k) < minorFrac * ref && Ek(k) < minorFrac * refE;   % no inhale and no exhale
    end
    % ---- step 2: a small pulse that only leads into a rest before the next inhale is a pause bump ----
    pk1 = nan(n0, 1);
    for k = 1:n0, pk1(k) = max(r(on(k):hiEnd(k))); end
    isLead = false(n0, 1);
    inh0 = find(~isMinor);
    for j = 2:numel(inh0) - 1
        k = inh0(j); kn = inh0(j + 1);
        a = on(k); b = max(a, on(kn) - 1);
        nxt = pk1(kn); m = j + 1;                       % highest pulse of the run the next inhale starts
        while m < numel(inh0)
            a2 = on(inh0(m)); b2 = max(a2, on(inh0(m + 1)) - 1);
            if areaE(a2, b2) >= exFrac * areaI(a2, b2), break; end
            m = m + 1; nxt = max(nxt, pk1(inh0(m)));
        end
        if areaE(a, b) >= exFrac * areaI(a, b) || pk1(k) >= leadFrac * nxt, continue; end
        if sum(abs(r1(a:b)) <= 2 * dS(a:b)) / fs >= restSec   % time at rest in its window (not "after its peak":
                                                               % the 1-s mean leaks the next inhale into the window end)
            isLead(k) = true; isMinor(k) = true;
        end
    end
    clear r1
    inh = find(~isMinor);                               % inhale rows (row 1 always)

    % ---- step 3: exhale since the previous inhale vs that inhale ----
    % Over [onset(k-1), onset(k)) of two consecutive inhale rows: I = the area above zero flow from onset(k-1) to the first
    % crossing from above to below zero flow (or to onset(k) when there is none); E = the area below zero flow from that crossing
    % to onset(k) (0 when there is no crossing: no exhale phase).  Inhale k joins when E < exFrac x I.  (Taking E and I as the
    % areas below / above zero flow over the whole window lets an onset sitting in an exhale trough count the rest of that trough
    % as its 'exhale', so it never joins the real inhale after it.)
    % (measurement switches, not for production: opts.step3Mode 'window' = the whole-window areas, opts.keepRule 'first' = keep-first only)
    s3win = strcmp(getOpt(opts, 'step3Mode', 'pairCross'), 'window'); keepFirstOnly = strcmp(getOpt(opts, 'keepRule', 'troughPause'), 'first');
    pairR = nan(n0, 1); pairI = nan(n0, 1); pairE = nan(n0, 1); pairX = nan(n0, 1);
    joinI = false(numel(inh), 1);
    for j = 2:numel(inh)
        a = on(inh(j - 1)); b = max(a, on(inh(j)) - 1);
        [I, E, cx] = pairAreas(r, a, b, s3win);
        pairI(inh(j - 1)) = I; pairE(inh(j - 1)) = E; pairX(inh(j - 1)) = cx;
        pairR(inh(j - 1)) = E / I;
        joinI(j) = E < exFrac * I;
    end
    if ~isempty(inh)
        a = on(inh(end)); b = hiEnd(n0);
        [I, E, cx] = pairAreas(r, a, b, s3win);
        pairI(inh(end)) = I; pairE(inh(end)) = E; pairX(inh(end)) = cx; pairR(inh(end)) = E / I;
    end
    % Which onset the joined breath keeps: the onset of the breath being joined (its first inhale, the group head) unless the head
    % has an exhale TROUGH immediately to its left and the joining inhale a flat PAUSE immediately to its left - then the head is the
    % sensor's rebound off the trough (the start of the pause, not an inhale): the head and everything after it up to the joining
    % inhale become the end of the breath BEFORE (its exhale / pause), and the joining inhale starts the breath.  Otherwise the
    % joining inhale is the one removed (keep-first).  On the 200-ms mean minus zero flow, local swing = median max - min of the
    % windows within +-10 rows:  trough = its minimum over the troughWinSec (1 s) before the onset is below zero flow by at least
    % troughFrac (0.25) x swing;  pause = over the pauseWinSec (0.5 s) that end pauseGapSec (0.1 s) before the onset the range stays
    % within pauseFrac (0.15) x swing and the mean within pauseLevelFrac (0.25) x swing of zero flow.  The first breath of the
    % recording always keeps its head.
    troughWin = max(1, round(getOpt(opts, 'troughWinSec', 1) * fs)); troughFrac = getOpt(opts, 'troughFrac', 0.25);
    pauseWin = max(1, round(getOpt(opts, 'pauseWinSec', 0.5) * fs)); pauseFrac = getOpt(opts, 'pauseFrac', 0.15);
    pauseLvl = getOpt(opts, 'pauseLevelFrac', 0.25);
    pauseGap = max(0, round(getOpt(opts, 'pauseGapSec', 0.1) * fs));   % the pause window ends this long before the onset
    swLoc = nan(n0, 1);
    for k = 1:n0, swLoc(k) = median(swing(max(1, k - nbhd):min(n0, k + nbhd)), 'omitnan'); end
    troughLeft = false(n0, 1); pauseLeft = false(n0, 1);
    for k = 1:n0
        o = on(k);
        wt = r(max(1, o - troughWin):o);
        troughLeft(k) = min(wt) <= -troughFrac * swLoc(k);
        wp = r(max(1, o - pauseGap - pauseWin):max(1, o - pauseGap));
        pauseLeft(k) = (max(wp) - min(wp)) <= pauseFrac * swLoc(k) && abs(mean(wp)) <= pauseLvl * swLoc(k);
    end
    grp = zeros(n0, 1); head = zeros(n0, 1); g = 0;
    isTail = false(n0, 1); isTroughHead = false(n0, 1); removalRatio = nan(n0, 1);
    jOf = zeros(n0, 1); jOf(inh) = 1:numel(inh);
    for k = 1:n0
        if isMinor(k), grp(k) = max(g, 1); continue; end
        j = jOf(k);
        if j == 1 || ~joinI(j), g = g + 1; grp(k) = g; head(g) = k; continue; end
        h = head(g); rr = pairR(inh(j - 1));
        if ~keepFirstOnly && g > 1 && troughLeft(h) && pauseLeft(k)
            mv = find(grp == g);                        % the head and the rows it holds become the end of the breath before
            grp(mv) = g - 1; isTail(mv) = true; isTroughHead(h) = true; removalRatio(h) = rr;
            grp(k) = g; head(g) = k;
        else
            grp(k) = g; removalRatio(k) = rr;           % keep-first: the joining inhale is removed
        end
    end

    % ---- build the merged rows ----
    nK = max([grp; 0]);
    keepRows = zeros(nK, 1); nP = zeros(nK, 1); nR = zeros(nK, 1); origRows = cell(nK, 1);
    inA = nan(nK, 1); exA = nan(nK, 1);
    for q = 1:nK
        mem = find(grp == q);
        kr = mem(1); last = mem(end);
        keepRows(q) = kr; nP(q) = sum(~isMinor(mem) & ~isTail(mem)); nR(q) = numel(mem); origRows{q} = mem';
        if numel(mem) > 1
            bmObj(kr, 5) = bmObj(last, 5); bmObj(kr, 6) = bmObj(last, 6);
            bmObj(kr, 7) = bmObj(kr, 6) - bmObj(kr, 2);
            pc = mem(~isTail(mem));                     % a trough rebound taken over as the end of this breath is not its inhale peak
            [~, ip] = max(x(pk(pc)) - z(pk(pc)));
            bmObj(kr, [3 4 9]) = bmObj(pc(ip), [3 4 9]);
            [~, it] = min(bmObj(mem, 10));
            bmObj(kr, [10 11]) = bmObj(mem(it), [10 11]);
            bmObj(kr, 8) = bmObj(kr, 3) - (bmObj(kr, 1) + bmObj(kr, 5)) / 2;
        end
        a = on(kr); b = hiEnd(last);
        inA(q) = areaI(a, b) / fs * 1000; exA(q) = areaE(a, b) / fs * 1000;
    end
    bmObj = bmObj(keepRows, :);
    bmObj(:, 14) = 1:nK;
    if isfield(bmFeatures, 'bmObjBreathIdx')
        bmFeatures.bmObjBreathIdx = bmFeatures.bmObjBreathIdx(keepRows);
        bmFeatures.nBreathsSegmented = nK;
    end
    zA = nan(n0, 1); if n0 >= 1, zA = z(on)'; end
    nMinor = sum(isMinor);
    info = struct('rule', sprintf(['(2026-10-29a) an inhale is not a new breath when the exhale since the previous inhale is < %g x ' ...
        'that inhale (inhale = area above zero flow from the previous onset to its first crossing below zero flow; exhale = ' ...
        'area below zero flow from that crossing to the onset, 0 without a crossing); the joined breath keeps the earlier ' ...
        'onset unless it has an exhale trough just before it and the joining inhale a flat pause just before it (then the ' ...
        'earlier onset ends the breath before and the joining inhale starts the breath); engine onsets whose window ' ...
        'holds neither an inhale nor an exhale (< %g x the local median beyond a +-%g x swing noise band), and small ' ...
        'pulses that only lead into a >= %g-s rest before the next inhale, join the breath before'], ...
        exFrac, minorFrac, deadFrac, restSec), 'version', 'mergeIncompleteExhales 2026-10-29a', ...
        'exFrac', exFrac, 'deadFrac', deadFrac, 'minorFrac', minorFrac, ...
        'zeroFlow', zInfo.method, 'zeroFlowFracDwell', zInfo.fracFromFlat, ...
        'nBefore', n0, 'nAfter', nK, 'nMinor', nMinor, 'nLeadBumps', sum(isLead), 'nMerged', n0 - nK - nMinor, ...
        'nPulses', nP, 'nRows', nR, 'inhaleArea', inA, 'exhaleArea', exA, 'exInRatio', exA ./ inA, ...
        'origRows', {origRows}, 'isMinor', isMinor, 'isLeadBump', isLead, 'pulseInhaleArea', Ik / fs * 1000, 'pairExInRatio', pairR, ...
        'engineOnsetSec', on0Sec, 'zeroFlowAtOnsets', zA, 'deadBand', dRow, ...
        'pairInhaleArea', pairI / fs * 1000, 'pairExhaleArea', pairE / fs * 1000, 'pairCrossSec', pairX / fs, ...
        'troughLeft', troughLeft, 'pauseLeft', pauseLeft, 'isTroughHead', isTroughHead, 'isTail', isTail, ...
        'removalRatio', removalRatio, 'nTroughHeads', sum(isTroughHead), ...
        'keepOpts', struct('troughWinSec', troughWin / fs, 'troughFrac', troughFrac, 'pauseWinSec', pauseWin / fs, ...
        'pauseFrac', pauseFrac, 'pauseLevelFrac', pauseLvl, 'pauseGapSec', pauseGap / fs, 'step3Mode', getOpt(opts, 'step3Mode', 'pairCross'), ...
        'keepRule', getOpt(opts, 'keepRule', 'troughPause')));
    if isfield(bmFeatures, 'conditioning')
        bmFeatures.conditioning.incompleteExhaleMerge = struct('rule', info.rule, 'exFrac', exFrac, ...
            'deadFrac', deadFrac, 'minorFrac', minorFrac, 'zeroFlow', info.zeroFlow, ...
            'nBefore', n0, 'nAfter', nK, 'nMinor', nMinor, 'nMerged', info.nMerged);
    end
end

function [I, E, cx] = pairAreas(r, a, b, s3win)
% inhale area I = sum of r > 0 from a to the first crossing from above to below zero flow; exhale area E = sum of -r (r < 0) from
% that crossing to b; no crossing: I over [a, b], E = 0.  cx = first sample at or below zero flow after the crossing (NaN = none)
    w = r(a:b);
    if s3win, I = sum(max(w, 0)); E = sum(max(-w, 0)); cx = NaN; return; end   % the whole-window rule (measurement only)
    i = find(w(1:end - 1) > 0 & w(2:end) <= 0, 1);
    if isempty(i)
        I = sum(max(w, 0)); E = 0; cx = NaN;
    else
        I = sum(max(w(1:i), 0)); E = sum(max(-w(i + 1:end), 0)); cx = a + i;
    end
end

function v = getOpt(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function [z, zi] = zeroFlowLevel(x, fs, halfWinSec, binFrac, sideFrac, useMax, massFrac)
% Zero-flow level of a nasal-pressure trace = its DWELL level: between breaths the pressure stays at zero flow (end-expiratory
% pauses, holds) while inhales go above it and exhales below it, so within +-halfWinSec (10 s) zero flow is the most frequent level
% of the trace (histogram mode of the 200-ms mean at 25 Hz, bins binFrac (0.04) x the local 5-95 % range, smoothed over 3 bins).
% A mode is accepted only when the trace spends >= sideFrac (0.10) of the window both above and below it (beyond 2 bins): the
% rounded trough of a slow exhale (or plateau of a slow inhale) without a pause is also a dwell level but the trace sits on one
% side of it only (a synthetic 4 /min trace without pauses would merge to one breath).  Elsewhere z = the 60-s moving mean
% (continuous breathing; a high-passed trace has its long-run mean near zero flow).  The accepted modes are median-filtered over
% 5 s.  Why a dwell level: a high-pass filter shifts the pause level after every breath (pauses sit above the long-run mean,
% with cardiac ripple on them, so they never read as 'flat'), and a pause's offset over several seconds adds up to a whole
% inhale's area.
    N = numel(x);
    xs = movmean(x, max(1, round(0.2 * fs)));
    dec = max(1, round(fs / 25));
    xd = xs(1:dec:end); fd = fs / dec; Nd = numel(xd);
    g = 1:fs:N;                                             % 1-s grid
    gd = min(Nd, round((g - 1) / dec) + 1);
    hw = round(halfWinSec * fd);
    zMean = movmean(x, round(60 * fs));
    zg = zMean(g); ok = false(size(g)); zm = nan(size(g));
    for i = 1:numel(g)
        w = xd(max(1, gd(i) - hw):min(Nd, gd(i) + hw));
        q = prctile(w, [5 95]); rg = q(2) - q(1);
        if ~(rg > 0), continue; end
        bw = binFrac * rg;
        edges = (min(w) - bw):bw:(max(w) + bw);
        if numel(edges) < 4, continue; end
        c = histcounts(w, edges);
        c = conv(c, [1 1 1] / 3, 'same');
        [~, im] = max(c);
        m = (edges(im) + edges(im + 1)) / 2;
        m = median(w(abs(w - m) <= bw));                    % refine within the peak (no bin-grid bias)
        up = mean(w > m + 2 * bw); dn = mean(w < m - 2 * bw); mass = mean(abs(w - m) <= bw);
        if up >= sideFrac && dn >= sideFrac && mass >= massFrac
            ok(i) = true; zm(i) = m;
        end
    end
    if any(ok)
        zf = zm; zf(ok) = movmedian(zm(ok), 5);               % over 5 accepted seconds
        if useMax, zg(ok) = max(zf(ok), zg(ok)); else, zg(ok) = zf(ok); end
    end
    z = interp1(g, zg, 1:N, 'linear', 'extrap');
    zi = struct('method', sprintf(['dwell level (histogram mode of the 200-ms mean within +-%g s, bins %g x the ' ...
        'local range, accepted when >= %g of the window lies above and below it), else the 60-s moving mean'], ...
        halfWinSec, binFrac, sideFrac), 'fracFromFlat', mean(ok));
end
