function [on2, pk2, tr2, info] = refineInhaleOnsets_zlp(rspStored, det, fs, on, pk, tr, opts) %#ok<INUSL>
%REFINEINHALEONSETS_ZLP  Re-place each inhale onset on the stored respiration trace.
%
%   [onsets2, peaks2, troughs2, info] = refineInhaleOnsets_zlp(rspStored, det, fs, onsets, peaks, troughs [, opts])
%
%   WHY.  The detection (prepBreathTrace_zlp 'conservative' + findInhaleOnsets_zlp 'kneeBacktrack') finds each onset on a
%   500-ms moving-mean, 30-s-normalized copy of the trace.  On the stored trace that onset is often EARLY (the trace is still
%   flat; the rise starts 0.1-0.3 s later), sometimes sits on a small BUMP well before a quick rise to the peak, and sometimes
%   on a small upward inflection RIDING HIGH on a larger inhale.  This step looks at the stored trace itself (lightly smoothed)
%   and moves an onset only when the trace shows a definitively better one; otherwise the detection's onset stays.
%
%   WHERE.  zlpEstimateAllFeatures calls it right after findInhaleOnsets_zlp (option 'refineOnsets', default true) and checks
%   what comes back.  Peaks and troughs are NOT changed and NO breath is added or dropped: peaks2 = troughs2 = [] (keep the
%   input), numel(onsets2) == numel(onsets), info.srcIdx = 1:n.
%
%   INPUTS.  rspStored = the trace the detection got (raw units, inhale positive, NaN-filled); det = the detection trace
%   (unused); fs (Hz); onsets / peaks / troughs = the detection's landmarks in samples.  Breath k = onsets(k), its peak p = the
%   FIRST peak after it, the previous peak pp = the last peak before it, its trough T = the last trough between pp and the onset.
%
%   SLOPE.  The stored trace is smoothed by moving means of scalesMs (40, 80, 160, 300 ms); slope = central difference over the
%   same span.  For each breath the FINEST scale whose steepest slope in the search range exceeds snrMin (5) x the local slope
%   noise is used (local noise = robust SD (1.4826 x MAD) of the slope's deviation from its own 0.5-s moving mean, within
%   +-noiseWinSec (10 s) of the onset).  Sharp sigh-like rises therefore use 40 ms, slow noisy breaths up to 300 ms.  Levels are
%   read from a 40-ms moving mean (levelSmoothMs).
%
%   FOOT OF A RISE.  In a range [lo, hi] before the peak p: the steepest rise is the sample maximizing
%   slope(t) x exp(-(p - t)/farTauSec) (farTauSec 2.5 s: of two similar rises, the one closer to the peak wins - regular breaths
%   have an onset, then a quick rise to the peak).  From it, walk back to the last sample where the slope has stayed below
%   max(footFrac (5 %) x the steepest slope, noiseSigmas (2) x noise) for dwellMs (50 ms): the foot.  The foot is moved forward
%   by biasFrac (0.2) x the slope scale (the moving mean smears the start of a rise back by about half its span).  If the slope
%   never stays low inside the range (the rise began before it) the next coarser scale is tried; if none works there is no foot.
%
%   RULES (in order; each later rule sees the result of the earlier ones).
%   1 'foot'   early-by-smoothing.  Range [onset - backMs (200 ms), p - minRiseMs (100 ms)], never before the floor (below).  The
%              foot replaces the onset when it moved by >= minShiftMs (20 ms).  A FORWARD move may step over a flat stretch or a
%              small bump, never over part of the rise itself.  The move is refused when the level at the foot is more than
%              skipRiseFrac (0.15) x the breath amplitude above the level at the detection's onset, or more than skipSteepFrac
%              (0.05) x the amplitude while the stepped-over stretch is steep (its 90th-percentile slope > skipSlopeFrac (0.45) x
%              the steepest slope of the new rise: the skipped part is itself an earlier stage of the rise).  Then the foot of an
%              EARLIER rise in the range (steepest slope >= earlierRiseFrac (25 %) of the first one) is tried instead (at most 3
%              times); if none passes, the detection's onset stays.
%   2 'far'    bump long before a quick rise.  An onset (after rule 1) farther from its peak than max(farMinSec (0.8 s), farMult
%              (1.5) x the median onset-to-peak time of the levelWinBreaths (8) neighbours on each side) is searched again in
%              [p - farMult x that median, p - minRiseMs]; the foot found there replaces it (forward moves only; the same checks
%              as rule 1, except that a SLOW stretch may skip up to farSkipFrac (0.40) x the amplitude of level).
%   3 'level'  onset riding on a larger inhale.  Reference level = the median onset level (after rules 1-2) of the WHOLE trace
%              handed to this function; margin = half the width of the middle levelBandFrac (30 %) of the raw trace values (65th
%              minus 35th percentile), i.e. the band of the usual onset level is that width centred on the median onset level.
%              (An earlier version used the median of the 8 neighbours on each side and 0.15 x the local amplitude: in noisy
%              stretches it moved onsets back by seconds.)  An onset above the band is searched again from the floor to
%              p - minRiseMs: the foot of the steepest rise; while that foot is still above the band, the foot of the earlier
%              rise (if its slope reaches earlierRiseFrac of the first); a foot more than lowMarginFrac (1.0) x the median
%              amplitude BELOW the reference is refused (an exhale dip or artefact, not this inhale's start).  When no foot gets
%              below the margin, the walk goes back to the last point at or below reference + margin and then to the foot of the
%              rise there.  Accepted when the new onset is lower by >= levelGainFrac (0.10) x amplitude and moved by
%              >= minShiftMs.
%
%   FLOOR / ALTERNATION.  Every search range starts no earlier than max(pp + guardMs (50 ms), T - troughTolMs (0 ms)): an onset
%   never moves into the previous exhale (before the pair's trough) or before the previous peak.  A final pass keeps every onset
%   strictly between the previous peak and its own peak and strictly ascending (rule tag '+clamp' when clamped; a collision
%   after rounding restores the detection's onset, rule '').  The detection's own floor (first 20 % of the trough->peak span on
%   the detection trace) is NOT re-imposed: on the stored trace the foot of a sharp rise can lie right after the trough.
%
%   AMPLITUDE.  Breath amplitude A(k) = level at p minus the lowest level in [floor, onset]; rule 3 uses the median of A over the
%   whole trace.
%
%   info (aligned with the returned onsets, which are the input onsets one-to-one):
%     srcIdx        1:n
%     rule          cellstr: '' (unchanged) | 'foot' | 'far' | 'level' (+ '+clamp')
%     shiftMs       refined - detection onset, ms (+ = later)
%     confidence    0..1 for a moved onset: (share of the amplitude left between the new onset and the peak) x (1 - slope at the
%                   new onset / steepest slope); 0 when unchanged
%     scaleMs       slope scale used (NaN when unchanged or for a level crossing without a foot)
%     level         40-ms-smoothed trace at the onset after rules 1-2 (rule 3: after it)
%     levelRef, levelMargin, ampLocal   rule 3's reference level (median onset level), margin (half the band width) and
%                   amplitude (median), repeated per breath; levelBand [lo hi], levelBandWidth, ampSession (scalars)
%     counts        struct: nFoot, nFar, nLevel, nClamp, nUnchanged
%     opts          the options used;  version  this function's version string
%     levelGuarded, levelPauseFloor, levelTroughRefused (per breath) and their counts, levelAmpNb, ampBreath
%
%   opts (struct, merged over the defaults; every constant above is a field): scalesMs snrMin noiseWinSec noiseSigmas biasFrac
%   levelSmoothMs backMs minRiseMs guardMs troughTolMs footFrac dwellMs farTauSec farMult farMinSec farSkipFrac minShiftMs
%   skipRiseFrac skipSlopeFrac skipSteepFrac levelWinBreaths levelBandFrac earlierRiseFrac levelGainFrac lowMarginFrac, and the
%   rule-3 guards levelGuard ('pause' = the default; 'rise' | 'none' are measurement options) levelGuardSec levelRiseFrac
%   levelPauseFrac levelPauseLevelFrac levelTroughFrac levelTroughWinSec (levelTroughFrac 0 = trough refusal off), levelRefMode
%   ('session' = the default; 'local' = the neighbour reference of the earlier version, measurement only) levelMarginFrac (used
%   by 'local' only).  The defaults are the validated configuration; the other values exist to reproduce earlier versions.
%
%   Base MATLAB only.
    version = 'refineInhaleOnsets_zlp 2026-10-29a';
    if nargin < 7 || isempty(opts), opts = struct(); end
    o = defaults(opts);
    x = double(rspStored(:)');
    if any(~isfinite(x))
        g = isfinite(x); idx = 1:numel(x);
        if nnz(g) >= 2, x(~g) = interp1(idx(g), x(g), idx(~g), 'linear', 'extrap'); else, x(~g) = 0; end
    end
    N = numel(x);
    on = double(on(:)'); pk = double(pk(:)'); tr = double(tr(:)');
    nOn = numel(on);
    pk2 = []; tr2 = [];
    on2 = on; rule = repmat({''}, 1, nOn); conf = zeros(1, nOn); scUsed = nan(1, nOn);
    if nOn == 0
        info = makeInfo(1:0, rule, zeros(1, 0), conf, scUsed, [], [], [], [], o, version);
        return
    end

    % ---- traces: lightly smoothed levels and multi-scale slopes ----
    s = movmean(x, max(1, round(o.levelSmoothMs / 1000 * fs)));
    nSc = numel(o.scalesMs);
    D = cell(1, nSc); HF = cell(1, nSc);
    for q = 1:nSc
        wq = max(1, round(o.scalesMs(q) / 1000 * fs)); hq = max(1, round(o.scalesMs(q) / 2000 * fs));
        sq = movmean(x, wq);
        dq = zeros(1, N); dq(hq + 1:N - hq) = (sq(2 * hq + 1:N) - sq(1:N - 2 * hq)) / (2 * hq / fs);
        D{q} = dq;
        HF{q} = abs(dq - movmean(dq, max(3, round(0.5 * fs))));
    end
    dwell = max(2, round(o.dwellMs / 1000 * fs));
    minRise = max(2, round(o.minRiseMs / 1000 * fs));
    back = round(o.backMs / 1000 * fs);
    guard = round(o.guardMs / 1000 * fs);
    tol = round(o.troughTolMs / 1000 * fs);
    minShift = o.minShiftMs / 1000 * fs;
    ctx = struct('s', {s}, 'D', {D}, 'fs', fs, 'o', o, 'dwell', dwell);

    % ---- per-breath structure ----
    P = nan(1, nOn); PP = nan(1, nOn); T = nan(1, nOn); LO = nan(1, nOn);
    for k = 1:nOn
        q = pk(pk > on(k)); if isempty(q), continue; end
        P(k) = q(1);
        r = pk(pk < on(k)); if isempty(r), PP(k) = 1; else, PP(k) = r(end); end
        u = tr(tr < on(k) & tr > PP(k)); if ~isempty(u), T(k) = u(end); end
        lo = PP(k) + guard;
        if isfinite(T(k)), lo = max(lo, T(k) - tol); end
        LO(k) = max(1, min(lo, on(k)));
    end
    valid = isfinite(P) & on >= 1 & on <= N;
    NZ = nan(nOn, nSc); nw = round(o.noiseWinSec * fs);
    for k = find(valid)
        a = max(1, round(on(k)) - nw); b = min(N, round(on(k)) + nw);
        for q = 1:nSc, NZ(k, q) = 1.4826 * median(HF{q}(a:5:b)); end
    end
    ctx.NZ = NZ;
    A = nan(1, nOn);
    for k = find(valid)
        A(k) = s(round(P(k))) - min(s(round(LO(k)):round(on(k))));
    end

    % ---- rule 1 'foot': foot of the steepest rise within [onset - backMs, peak - minRise] ----
    for k = find(valid)
        ok = round(on(k)); p = round(P(k));
        lo = max(round(LO(k)), ok - back); hi = p - minRise;
        if hi <= lo + dwell, continue; end
        [f, dm, sq] = footChecked(ctx, k, lo, hi, p, ok, o.skipRiseFrac, max(A(k), eps));
        if ~isfinite(f) || abs(f - ok) < minShift, continue; end
        on2(k) = f; rule{k} = 'foot'; scUsed(k) = o.scalesMs(sq); conf(k) = confidenceOf(s, D{sq}, f, p, dm, A(k));
    end

    % ---- rule 2 'far': onsets far from their peak compared with the neighbours ----
    rise = nan(1, nOn); for k = find(valid), rise(k) = (P(k) - on2(k)) / fs; end
    riseRef = localMedian(rise, o.levelWinBreaths, false);
    for k = find(valid & rise > max(o.farMinSec, o.farMult * riseRef))
        ok = round(on2(k)); p = round(P(k));
        lo = max(round(LO(k)), round(p - o.farMult * riseRef(k) * fs)); hi = p - minRise;
        if hi <= lo + dwell || lo <= ok, continue; end
        [f, dm, sq] = footChecked(ctx, k, lo, hi, p, ok, o.farSkipFrac, max(A(k), eps));
        if ~isfinite(f) || f - ok < minShift, continue; end
        on2(k) = f; rule{k} = 'far'; scUsed(k) = o.scalesMs(sq); conf(k) = confidenceOf(s, D{sq}, f, p, dm, A(k));
    end

    % ---- rule 3 'level': onsets riding high on a larger inhale ----
    % The usual onset level is a whole-trace band, not the median of the neighbouring onsets (in a noisy stretch the neighbours'
    % onsets sit far from the usual level and a local reference sent onsets back by seconds).  Band width = the range of the
    % middle levelBandFrac (30 %) of the raw trace values (35th to 65th percentile of the whole trace handed to this function),
    % centred on the median onset level (after rules 1-2).  An onset above the band is searched again; a foot more than
    % lowMarginFrac x the median amplitude below the band centre is refused.
    % Pause floor (levelGuard 'pause'): when the detection's onset itself follows a flat pause, rule 3 may still pull an onset
    % that rules 1-2 pushed up the rise back towards it, but never before it (a sigh-like rise whose second stage rule 2 took
    % for the onset would otherwise be searched back into the pause).
    % Guard (levelRiseFrac > 0, levelGuard 'rise'): the rule moves only an onset that RIDES on a rise - the level climbs into it
    % by more than levelRiseFrac x the amplitude over the levelGuardSec (0.5 s) before it.  An onset after a flat pause or a
    % falling stretch is the start of an inhale even when it sits above the band, and is left to rules 1-2.  levelRefMode
    % 'local' = the neighbour reference of the earlier version (8 neighbours each side, margin levelMarginFrac x the local
    % amplitude), for comparison.
    lvl = nan(1, nOn);
    for k = find(valid), lvl(k) = s(round(on2(k))); end
    medOn = medianFinite(lvl(valid));
    bandW = middleRange(x, o.levelBandFrac);
    Asess = medianFinite(A);
    if strcmp(o.levelRefMode, 'local')
        Aloc = localMedian(A, o.levelWinBreaths, true); Aloc(~isfinite(Aloc)) = Asess;
        lvlRef = localMedian(lvl, o.levelWinBreaths, false);
        margin = o.levelMarginFrac * Aloc;
    else
        assert(strcmp(o.levelRefMode, 'session'), 'refineInhaleOnsets_zlp:levelRefMode', 'levelRefMode must be session or local');
        lvlRef = repmat(medOn, 1, nOn);
        margin = repmat(bandW / 2, 1, nOn);
        Aloc = repmat(Asess, 1, nOn);
    end
    gWin = max(1, round(o.levelGuardSec * fs));
    guarded = false(1, nOn); troughRefused = false(1, nOn); pauseFloor = false(1, nOn); tWin = max(1, round(o.levelTroughWinSec * fs));
    AlocNb = localMedian(A, o.levelWinBreaths, true); AlocNb(~isfinite(AlocNb)) = Asess;   % breath size near the onset (pause guard)
    for k = find(valid & lvl > lvlRef + margin)
        ok = round(on2(k)); p = round(P(k));
        lo = round(LO(k)); hi = p - minRise;
        if strcmp(o.levelGuard, 'pause')         % the ENGINE onset follows a pause: rule 3 may pull a pushed-up onset back, but not before it
            we = s(max(1, round(on(k)) - gWin):round(on(k)));
            if max(we) - min(we) <= o.levelPauseFrac * AlocNb(k) && abs(mean(we) - medOn) <= o.levelPauseLevelFrac * AlocNb(k)
                lo = max(lo, round(on(k))); pauseFloor(k) = true;
            end
        end
        if hi <= lo + dwell, continue; end
        wg = s(max(1, ok - gWin):ok);
        switch o.levelGuard
            case 'rise'    % move only an onset the level climbs into
                if s(ok) - min(wg) <= o.levelRiseFrac * Aloc(k), guarded(k) = true; continue; end
            case 'pause'   % do not move an onset that follows a flat pause near the usual onset level
                if max(wg) - min(wg) <= o.levelPauseFrac * AlocNb(k) && abs(mean(wg) - medOn) <= o.levelPauseLevelFrac * AlocNb(k)
                    guarded(k) = true; continue
                end
            case 'none'
            otherwise, error('refineInhaleOnsets_zlp:levelGuard', 'levelGuard must be rise, pause or none');
        end
        thrLvl = lvlRef(k) + margin(k);
        lowLvl = lvlRef(k) - o.lowMarginFrac * Aloc(k);
        [f, dm, sq] = footLevel(ctx, k, lo, hi, p, thrLvl, lowLvl, A(k));
        if ~isfinite(f), continue; end
        if o.levelTroughFrac > 0 && medOn - min(s(max(1, f - tWin):f)) >= o.levelTroughFrac * AlocNb(k)
            troughRefused(k) = true; continue      % the new onset would come out of an exhale trough: a rebound, not the inhale start
        end
        if s(f) < s(ok) - o.levelGainFrac * max(A(k), eps) && abs(f - ok) >= minShift
            on2(k) = f; rule{k} = 'level'; lvl(k) = s(f);
            if isfinite(sq), scUsed(k) = o.scalesMs(sq); conf(k) = confidenceOf(s, D{sq}, f, p, dm, A(k)); else, scUsed(k) = NaN; conf(k) = 0.5; end
        end
    end

    % ---- alternation: strictly between the previous peak and the own peak, strictly ascending ----
    lastOn = -inf;
    for k = 1:nOn
        lowB = max(max(lastOn + 1, 1), round(PP(k)) + 1); if ~isfinite(lowB), lowB = max(lastOn + 1, 1); end
        upB = N; if isfinite(P(k)), upB = P(k) - 1; end
        if on2(k) < lowB || on2(k) > upB
            on2(k) = min(max(on2(k), lowB), upB);
            if round(on2(k)) == round(on(k)), rule{k} = ''; conf(k) = 0; else, rule{k} = [rule{k} '+clamp']; end
        end
        lastOn = on2(k);
    end
    on2 = round(on2);
    for k = 2:nOn
        if on2(k) <= on2(k - 1), on2(k) = round(on(k)); rule{k} = ''; conf(k) = 0; end
    end
    unchanged = on2 == round(on);
    rule(unchanged) = {''}; conf(unchanged) = 0; scUsed(unchanged) = NaN;
    shiftMs = (on2 - on) / fs * 1000;
    info = makeInfo(1:nOn, rule, shiftMs, conf, scUsed, lvl, lvlRef, margin, Aloc, o, version);
    info.levelBand = [medOn - bandW / 2, medOn + bandW / 2];
    info.levelGuarded = guarded; info.nLevelGuarded = nnz(guarded);
    info.levelAmpNb = AlocNb; info.ampBreath = A; info.levelPauseFloor = pauseFloor; info.nLevelPauseFloor = nnz(pauseFloor);
    info.levelTroughRefused = troughRefused; info.nLevelTroughRefused = nnz(troughRefused);
    info.levelBandWidth = bandW;
    info.ampSession = Asess;
end

function w = middleRange(x, frac)
% width of the middle frac of the values of x (frac 0.30: 65th minus 35th percentile; nearest rank, base MATLAB)
    v = sort(x(isfinite(x)));
    if isempty(v), w = 0; return; end
    n = numel(v);
    lo = v(min(n, max(1, ceil((0.5 - frac / 2) * n))));
    hi = v(min(n, max(1, ceil((0.5 + frac / 2) * n))));
    w = hi - lo;
end

% ------------------------------------------------------------------------------------------------
function o = defaults(u)
    o = struct( ...
        'levelSmoothMs', 40, ...                         % levels are read from this moving mean
        'scalesMs', [40 80 160 300], ...                 % slope scales (moving mean and central-difference span)
        'snrMin', 5, ...                                 % a scale is usable when its steepest slope >= snrMin x local slope noise
        'noiseWinSec', 10, 'noiseSigmas', 2, ...         % noise window (+- s); the foot slope threshold is at least noiseSigmas x noise
        'biasFrac', 0.2, ...                             % foot moved forward by biasFrac x scale (moving-mean footprint)
        'backMs', 200, ...                               % rule 1 searches from onset - backMs ...
        'minRiseMs', 100, ...                            % ... to peak - minRiseMs
        'guardMs', 50, ...                               % never closer than this after the previous peak
        'troughTolMs', 0, ...                            % never earlier than the pair's trough minus this
        'footFrac', 0.05, 'dwellMs', 50, ...             % foot: slope < footFrac x steepest for dwellMs
        'farTauSec', 2.5, ...                            % proximity weight exp(-(p - t)/farTauSec) of the steepest rise
        'farMult', 1.5, 'farMinSec', 0.8, 'farSkipFrac', 0.40, ...   % rule 2
        'minShiftMs', 20, ...                            % smaller moves are not made
        'skipRiseFrac', 0.15, ...                        % a forward move may skip at most this x amplitude of level ...
        'skipSlopeFrac', 0.45, ...                       % ... and at most skipSteepFrac x amplitude when the skipped stretch is steep
        'skipSteepFrac', 0.05, ...                       %     (90th-pct slope > skipSlopeFrac x the new rise's steepest slope)
        'earlierRiseFrac', 0.25, ...                     % an earlier rise counts when its slope >= this x the steepest
        'levelWinBreaths', 8, ...                          % rule 2: neighbours each side for the median rise time
        'levelBandFrac', 0.30, 'levelGainFrac', 0.10, 'lowMarginFrac', 1.0, ...  % rule 3 (whole-trace band)
        'levelRefMode', 'session', 'levelMarginFrac', 0.15, ...   % 'local' = the neighbour reference of the earlier version (for comparison)
        'levelGuard', 'pause', 'levelGuardSec', 0.5, ...  % rule 3 guard: 'pause' | 'rise' | 'none' (= the band alone)
        'levelRiseFrac', 0.10, 'levelPauseFrac', 0.15, 'levelPauseLevelFrac', 0.25, ...
        'levelTroughFrac', 0.35, 'levelTroughWinSec', 1.0);   % rule 3 refuses a new onset with an exhale trough this deep (x the local
                                                              % amplitude below the band centre) within this time before it (0 = off)
    if isstruct(u)
        fn = fieldnames(u);
        for i = 1:numel(fn)
            assert(isfield(o, fn{i}), 'refineInhaleOnsets_zlp:unknownOption', 'unknown option %s', fn{i});
            o.(fn{i}) = u.(fn{i});
        end
    end
end

function info = makeInfo(src, rule, shiftMs, conf, scUsed, lvl, lvlRef, margin, Aloc, o, version)
    cnt = struct('nFoot', nnz(strncmp(rule, 'foot', 4)), 'nFar', nnz(strncmp(rule, 'far', 3)), 'nLevel', nnz(strncmp(rule, 'level', 5)), ...
        'nClamp', nnz(contains(rule, 'clamp')), 'nUnchanged', nnz(cellfun(@isempty, rule)));
    info = struct('srcIdx', src, 'rule', {rule}, 'shiftMs', shiftMs, 'confidence', conf, 'scaleMs', scUsed, 'level', lvl, ...
        'levelRef', lvlRef, 'levelMargin', margin, 'ampLocal', Aloc, 'counts', cnt, 'opts', o, 'version', version);
end

function [f, dm, sq] = footMulti(ctx, k, lo, hi, p)
% foot of the steepest (proximity-weighted) rise in [lo, hi] at the finest usable slope scale; NaN = none
    o = ctx.o; fs = ctx.fs; f = NaN; dm = NaN; sq = NaN;
    seg = lo:hi; w = exp(-max(0, p - seg) / fs / o.farTauSec);
    for q = 1:numel(o.scalesMs)
        dq = ctx.D{q}; nz = ctx.NZ(k, q);
        [v, j] = max(dq(seg) .* w);
        if ~(v > 0), continue; end
        m = seg(j); dmq = dq(m);
        if dmq < o.snrMin * nz, continue; end
        thr = max(o.footFrac * dmq, o.noiseSigmas * nz);
        low = dq(lo:m) < thr;
        okRun = movsum(double(low), [ctx.dwell - 1 0]) == ctx.dwell;
        okRun(1:min(numel(okRun), ctx.dwell - 1)) = false;
        idx = find(okRun, 1, 'last');
        if isempty(idx), continue; end
        f = lo - 1 + idx;
        f = min(m - 1, f + round(o.biasFrac * o.scalesMs(q) / 1000 * fs));
        dm = dmq; sq = q; return
    end
end

function [f, dm, sq] = footChecked(ctx, k, lo, hi, p, ok, skipFrac, Ak)
% foot of the steepest rise; a forward foot that skips more than skipFrac x Ak of level above the onset ok (or more than
% skipSteepFrac x Ak over a steep stretch) is replaced by the foot of an earlier rise (slope >= earlierRiseFrac of the steepest) up to 3 times, else NaN (keep the engine onset)
    o = ctx.o; s = ctx.s;
    [f, dm, sq] = footMulti(ctx, k, lo, hi, p);
    if ~isfinite(f), return; end
    dm0 = dm; tries = 0;
    while f > ok && ((s(f) - s(ok)) > skipFrac * Ak || ((s(f) - s(ok)) > o.skipSteepFrac * Ak && skippedSlope(ctx.D{sq}, ok, f) > o.skipSlopeFrac * dm))
        e = f - ctx.dwell;
        if tries >= 3 || e <= lo + 2 * ctx.dwell, f = NaN; dm = NaN; sq = NaN; return; end
        [f2, d2, q2] = footMulti(ctx, k, lo, e, p);
        if ~isfinite(f2) || d2 < o.earlierRiseFrac * dm0, f = NaN; dm = NaN; sq = NaN; return; end
        f = f2; dm = d2; sq = q2; tries = tries + 1;
    end
end

function v = skippedSlope(d, a, b)
% 90th percentile (nearest rank) of the slope over the stretch [a, b) a forward move steps over
    if b <= a, v = -inf; return; end
    w = sort(d(a:b - 1));
    v = w(min(numel(w), max(1, ceil(0.9 * numel(w)))));
end

function [f, dm, sq] = footLevel(ctx, k, lo, hi, p, thrLvl, lowLvl, A)
% rule 3: foot of the steepest rise; while above thrLvl the foot of the earlier rise (never below lowLvl); else the last level
% crossing at or below thrLvl, refined to the foot of the rise there
    s = ctx.s; o = ctx.o; f = NaN; dm = NaN; sq = NaN; dm0 = NaN; e = hi; dwell = ctx.dwell;
    for it = 1:5
        [f1, d1, q1] = footMulti(ctx, k, lo, e, p);
        if ~isfinite(f1) || s(f1) < lowLvl, break; end
        if it == 1, dm0 = d1; end
        f = f1; dm = d1; sq = q1;
        if s(f) <= thrLvl, return; end
        e = f1 - dwell;
        if e <= lo + 2 * dwell, break; end
        if max(ctx.D{q1}(lo:e)) < o.earlierRiseFrac * dm0, break; end
    end
    if ~isfinite(f), f = hi; dm = NaN; sq = NaN; end
    c = find(s(lo:f) <= thrLvl & s(lo:f) >= lowLvl, 1, 'last');
    if ~isempty(c)
        c = lo - 1 + c;
        [f2, d2, q2] = footMulti(ctx, k, lo, min(f, c + round(0.15 * ctx.fs)), p);
        if isfinite(f2) && f2 <= c && s(f2) >= lowLvl, f = f2; dm = d2; sq = q2; else, f = c; dm = NaN; sq = NaN; end
    else
        [vmin, j] = min(s(lo:f));
        if vmin >= lowLvl && vmin < s(f) - 0.05 * max(A, eps), f = lo - 1 + j; dm = NaN; sq = NaN; end
    end
end

function c = confidenceOf(s, d, f, p, dm, A)
    if ~isfinite(dm) || dm <= 0, c = 0; return; end
    fr = min(1, max(0, (s(p) - s(f)) / max(A, eps)));
    fl = min(1, max(0, 1 - max(d(f), 0) / dm));
    c = fr * fl;
end

function m = localMedian(v, W, includeSelf)
    n = numel(v); m = nan(1, n);
    for k = 1:n
        a = max(1, k - W); b = min(n, k + W);
        w = v(a:b); if ~includeSelf, w(k - a + 1) = NaN; end
        w = w(isfinite(w));
        if ~isempty(w), m(k) = median(w); end
    end
end

function v = medianFinite(a)
    a = a(isfinite(a)); if isempty(a), v = 0; else, v = median(a); end
end
