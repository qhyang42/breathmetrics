function [bm, rawVolumes, zlpInfo] = zlpEstimateAllFeatures(resp, srate, varargin)
%ZLPESTIMATEALLFEATURES  breathmetrics features from ZLP breath segmentation.
%
%   bm = zlpEstimateAllFeatures(resp, srate)
%   [bm, rawVolumes, zlpInfo] = zlpEstimateAllFeatures(resp, srate, 'Name', value, ...)
%
%   Drop-in alternative to bm.estimateAllFeatures() for recordings where the
%   stock extrema / inhale-onset detection struggles (amplitude-non-
%   stationary signals, drifting baselines, paced or very slow breathing,
%   belt/pressure/cannula sensors). Segmentation is replaced by the ZLP
%   (Zelano Lab Preprocessing) engine - validity-ruled extrema
%   (prepBreathTrace_zlp) and a slope-walk "kneeBacktrack" inhale onset
%   (findInhaleOnsets_zlp) - while EVERY downstream feature (exhale onsets,
%   pauses, offsets, durations, volumes, secondary statistics) is still
%   computed by the unmodified breathmetrics class: the landmarks are
%   injected and manualAdjustPostProcess(), the class' own post-manual-edit
%   hook, recomputes the feature set from them. Full algorithm specification
%   and motivation: docs/zlp_segmentation.md.
%
%   Two stages follow the detection (both documented in
%   docs/zlp_segmentation.md):
%     onset refinement (ON by default, 'refineOnsets'): refineInhaleOnsets_zlp
%       re-places each inhale onset on the stored trace itself - the foot of
%       the rise - where the detection's onset is early, sits on a bump, or
%       rides high on a larger inhale. Peaks, troughs and the number of
%       breaths are unchanged. 'refineOnsets', false gives exactly the
%       landmarks (and so the bm object) of the earlier version of this
%       function.
%     incomplete-exhale merge (OFF by default, 'mergeIncompleteExhales'):
%       mergeIncompleteExhales_zlp joins an inhale to the breath before it when
%       there has been no exhale since that breath's inhale (double inhales,
%       staircase inhales, pause blips). The joined landmarks are injected, so
%       the class recomputes EVERY feature (durations, volumes, pauses, shape
%       features, secondary statistics) for each whole merged breath: the
%       object holds one breath per merged breath. (A table that keeps the
%       per-breath features of the detection's rows and only drops the joined
%       rows would describe the first pulse of a merged breath; this function
%       does not do that.)
%
%   INPUTS
%   resp  : respiration trace, numeric vector, inhale positive, RAW units.
%           NaNs are tolerated (filled internally for detection).
%   srate : sampling rate (Hz).
%
%   Name-value options
%   'dataType'       ('humanAirflow') passed to the breathmetrics
%                    constructor. Only the airflow data types are supported
%                    (the manual-adjust path exists only for them).
%   'floorFrac'      (0.05) amplitude-normalization floor, as a fraction of
%                    the median 30-s moving std - stops near-silent
%                    stretches being amplified into fake breaths.
%   'blankBelowFrac' ([]) zero the detection copy where the local amplitude
%                    scale is below this fraction of its median (hardware-
%                    attenuated stretches). [] = off.
%   'sighSpan'       ([]) [startSample endSample] of a paced-sigh /
%                    double-inhale block; inside it, of two detected peaks
%                    closer than 5 s only the first is kept, so the double
%                    inhale segments as one breath. [] = off.
%   'verbose'        (0) passed to the breathmetrics methods.
%   'refineOnsets'   (true) re-place the inhale onsets on the stored trace
%                    right after the detection (refineInhaleOnsets_zlp). The
%                    result is checked (strictly ascending, one per detected
%                    onset, each with its own peak after it, no two onsets
%                    sharing a peak); a violation is an error
%                    (zlpEstimateAllFeatures:inconsistentRefine). false = the
%                    onsets of the detection as they were.
%   'refineOpts'     (struct()) options struct of refineInhaleOnsets_zlp
%                    (unknown fields error).
%   'mergeIncompleteExhales' (false) join inhales that have no exhale since the
%                    previous breath's inhale (mergeIncompleteExhales_zlp, run
%                    on the paired breaths 1..n-1 after the onset refinement)
%                    and compute the features of the merged breaths.
%   'mergeExFrac'    (0.20) an inhale joins the breath before when the exhale
%                    since the previous inhale is under this fraction of it.
%   'mergeOpts'      (struct()) options struct of mergeIncompleteExhales_zlp.
%                    If a merged breath's lowest member trough does not lie after
%                    its merged peak, the lowest member trough that does is
%                    used (counted in zlpInfo.merge.nTroughGuard).
%
%   OUTPUTS
%   bm         : a breathmetrics object whose inhalePeaks / exhaleTroughs /
%                inhaleOnsets are the ZLP landmarks and whose remaining
%                features were recomputed by the class from them.
%                featuresManuallyEdited is set to 1 so downstream code knows
%                the landmarks were injected. NOTE the object's respiration
%                is the windowed-NORMALIZED detection trace, so its
%                flow/volume features are in locally-rescaled units - see
%                rawVolumes below and the units section of the doc.
%   rawVolumes : (optional) struct with raw-signal-unit volume twins,
%                computed only when requested:
%                .inhaleVolumesRaw / .exhaleVolumesRaw - the toolbox's own
%                volume integral (findRespiratoryVolumes) evaluated on the
%                RAW-unit trace (60-s moving-mean baseline removed) between
%                the SAME onset/offset landmarks stored in bm. The windowed
%                normalization that makes detection robust also makes the
%                object's volumes incomparable across epochs whose breathing
%                depth differs (the 30-s moving-std divisor tracks local
%                depth, so physically large breaths in deep-breathing
%                stretches read SMALLER); these twins restore interpretable
%                amplitude while leaving every timing metric untouched.
%                .note documents the convention. Units remain arbitrary
%                sensor units (not liters): within-recording comparisons are
%                meaningful; normalize per recording for between-subject use.
%   zlpInfo    : (optional) struct with the records of the two new stages:
%                .refine - the check record of the onset refinement (empty
%                  when 'refineOnsets' is false): onsetsIn / peaksIn / troughsIn
%                  (the detection's landmarks, samples), onsetsOut / peaksOut /
%                  troughsOut (what went into the pairing; peaks and troughs are
%                  unchanged), nOnIn, nOnOut, nPeaksIn, nPeaksOut,
%                  nOrphanPeaksIn, nOrphanPeaksOut, rule (cellstr aligned with
%                  onsetsOut: '' = detection onset kept, else 'foot' | 'far' |
%                  'level', + '+clamp'), shiftMs (refined - detected onset, ms)
%                  and refineFn (refineInhaleOnsets_zlp's own info). onsetsOut
%                  includes the last detected onset and onsets dropped by the
%                  landmark pairing.
%                .merge - empty unless 'mergeIncompleteExhales' is true, else
%                  a struct with .merged and .info (the outputs of
%                  mergeIncompleteExhales_zlp, on the paired breaths 1..n-1) and
%                  .nTroughGuard.
%
%   REQUIREMENTS: base MATLAB R2017b+ (movmean/movstd/movmax/movmin/movsum,
%   fillmissing), the Signal Processing Toolbox (findpeaks, used by the
%   'conservative' extrema mode) and, for the breath merge, prctile
%   (Statistics and Machine Learning Toolbox).
%
%   EXAMPLE
%       bmDat = load('sample_data.mat');          % ships with the toolbox
%       [bm, raw] = zlpEstimateAllFeatures(bmDat.resp, bmDat.srate);
%       bm.plotFeatures({'extrema', 'onsets'});
%       median(raw.inhaleVolumesRaw, 'omitnan')
%
%       % earlier behaviour (no onset refinement), or with the breath merge
%       bm0 = zlpEstimateAllFeatures(bmDat.resp, bmDat.srate, 'refineOnsets', false);
%       [bm1, ~, zi] = zlpEstimateAllFeatures(bmDat.resp, bmDat.srate, ...
%                          'mergeIncompleteExhales', true);
%       zi.merge.info.nMerged

    p = inputParser;
    p.addParameter('dataType', 'humanAirflow');
    p.addParameter('floorFrac', 0.05);
    p.addParameter('blankBelowFrac', []);
    p.addParameter('sighSpan', []);
    p.addParameter('verbose', 0);
    p.addParameter('refineOnsets', true);
    p.addParameter('refineOpts', struct());
    p.addParameter('mergeIncompleteExhales', false);
    p.addParameter('mergeExFrac', 0.20);
    p.addParameter('mergeOpts', struct());
    p.parse(varargin{:});
    opt = p.Results;

    if ~ismember(opt.dataType, {'humanAirflow', 'rodentAirflow'})
        error('zlpEstimateAllFeatures:dataType', ...
            ['dataType must be humanAirflow or rodentAirflow - the ' ...
             'manual-adjust path this function relies on exists only for ' ...
             'airflow data types.']);
    end

    if size(resp, 1) > 1, resp = resp'; end
    assert(isvector(resp) && isnumeric(resp), 'resp must be a numeric vector');
    respFilled = fillmissing(double(resp), 'linear', 'EndValues', 'nearest');

    % ---- ZLP segmentation (locked production configuration) ----
    [det, pk0, tr0] = prepBreathTrace_zlp(respFilled, srate, 'conservative', ...
        opt.blankBelowFrac, opt.sighSpan, opt.floorFrac);
    [on, pkP, trP] = findInhaleOnsets_zlp(det, srate, pk0, tr0, ...
        'kneeBacktrack', 0.4, 1.25, 0.50, 0.10);
    zlpInfo = struct('refine', [], 'merge', []);
    if opt.refineOnsets
        % re-place the onsets on the stored trace (refineInhaleOnsets_zlp)
        [on, pkP, trP, zlpInfo.refine] = applyOnsetRefinement(respFilled, det, ...
            srate, on, pkP, trP, opt.refineOpts);
    end
    on = round(sort(on(:)'));
    assert(numel(on) >= 3, 'zlpEstimateAllFeatures:tooFewBreaths', ...
        'only %d inhale onsets detected - not a usable respiration trace', numel(on));

    % breathmetrics-convention per-inhale landmark arrays: inhalePeaks(k) =
    % the pair peak after onset k; exhaleTroughs(k) = the first trough after
    % that peak (strict alternation makes it the trough inside breath k).
    % Trailing inhales with no following trough are dropped (the class'
    % simplify convention).
    n0 = numel(on);
    pkA = nan(1, n0); trA = nan(1, n0);
    for k = 1:n0
        pcand = pkP(pkP > on(k));
        if isempty(pcand), continue; end
        pkA(k) = pcand(1);
        tcand = trP(trP > pkA(k));
        if ~isempty(tcand), trA(k) = tcand(1); end
    end
    keep = isfinite(pkA);
    on = on(keep); pkA = pkA(keep); trA = trA(keep);
    while ~isempty(on) && ~isfinite(trA(end))
        on(end) = []; pkA(end) = []; trA(end) = [];
    end
    n = numel(on);
    assert(n >= 3, 'zlpEstimateAllFeatures:tooFewBreaths', ...
        'only %d complete breaths after landmark pairing', n);
    pkA = round(pkA); trA = round(trA);

    % ---- optional: join inhales that have no exhale since the previous one ----
    % Judged on breaths 1..n-1 (windows [on(k), on(k+1)), the rows of a breath
    % table); the last paired inhale has no closing onset and is kept as it is.
    % The merged landmarks replace the originals below, so the class computes
    % every feature for each whole merged breath.
    if opt.mergeIncompleteExhales
        [mg, mi] = mergeIncompleteExhales_zlp(respFilled, srate, on, pkA(1:n-1), ...
            trA(1:n-1), opt.mergeExFrac, opt.mergeOpts);
        rawBC0 = respFilled - movmean(respFilled, round(60 * srate));
        mTr = mg.troughs; nTroughGuard = 0;
        for q = 1:numel(mg.onsets)
            if mTr(q) <= mg.peaks(q)
                mem = mg.origRows{q};
                cand = trA(mem); cand = cand(cand > mg.peaks(q));
                [~, ic] = min(rawBC0(cand));
                mTr(q) = cand(ic);
                nTroughGuard = nTroughGuard + 1;
            end
        end
        mg.troughs = mTr;
        on = [mg.onsets, on(n)];
        pkA = [mg.peaks, pkA(n)];
        trA = [mg.troughs, trA(n)];
        assert(all(on(1:end-1) < pkA(1:end-1) & pkA(1:end-1) < trA(1:end-1) & ...
            trA(1:end-1) < on(2:end)), 'zlpEstimateAllFeatures:inconsistentMerge', ...
            'merged landmarks are not in onset < peak < trough < next onset order');
        zlpInfo.merge = struct('merged', mg, 'info', mi, 'nTroughGuard', nTroughGuard);
        n = numel(on);
        assert(n >= 3, 'zlpEstimateAllFeatures:tooFewBreaths', ...
            'only %d breaths after the merge', n);
    end

    % ---- inject the landmarks; the class computes everything else ----
    % Sanctioned manual-adjust path: set extrema, let the class derive its
    % own exhale-onset / pause estimates from them (findOnsetsAndPauses),
    % replace the inhale onsets with the ZLP ones, drop any pause onset the
    % new onsets contradict, then manualAdjustPostProcess() recomputes
    % offsets / durations / volumes / secondary features downstream of the
    % adjusted landmarks - exactly what the GUI does after a manual edit.
    bm = breathmetrics(det, srate, opt.dataType);
    bm.correctRespirationToBaseline('sliding', 0, opt.verbose);
    respBC = bm.baselineCorrectedRespiration(:)';
    bm.inhalePeaks   = pkA;
    bm.exhaleTroughs = trA;
    bm.peakInspiratoryFlows  = respBC(pkA);
    bm.troughExpiratoryFlows = respBC(trA);
    bm.findOnsetsAndPauses(opt.verbose);
    bm.inhaleOnsets = on;
    % consistency guard: a pause onset the class estimated from ITS onsets
    % can contradict the injected ones - an inhale pause must fall strictly
    % between the breath's peak and its exhale onset, an exhale pause
    % strictly between the trough and the NEXT inhale onset. Contradicting
    % pauses are set to NaN (= no pause), the class' own missing-pause code.
    inP = bm.inhalePauseOnsets; exP = bm.exhalePauseOnsets;
    exhOn = bm.exhaleOnsets;
    for bi = 1:n
        if bi <= numel(inP) && ~isnan(inP(bi)) && bi <= numel(exhOn) && ...
                ~isnan(exhOn(bi)) && (inP(bi) <= pkA(bi) || inP(bi) >= exhOn(bi))
            inP(bi) = NaN;
        end
        if bi <= numel(exP) && ~isnan(exP(bi)) && (exP(bi) <= trA(bi) || ...
                (bi + 1 <= n && exP(bi) >= on(bi + 1)))
            exP(bi) = NaN;
        end
    end
    bm.inhalePauseOnsets = inP;
    bm.exhalePauseOnsets = exP;
    bm.inhaleTimeToPeak = (pkA - on) / srate;
    bm.manualAdjustPostProcess();
    bm.featuresManuallyEdited = 1;

    % ---- raw-signal-unit volume twins (on request) ----
    if nargout > 1
        rawBC = respFilled - movmean(respFilled, round(60 * srate));
        [vIn, vEx] = findRespiratoryVolumes(rawBC, srate, ...
            bm.inhaleOnsets, bm.exhaleOnsets, bm.inhaleOffsets, bm.exhaleOffsets);
        rawVolumes = struct( ...
            'inhaleVolumesRaw', vIn, ...
            'exhaleVolumesRaw', vEx, ...
            'note', ['findRespiratoryVolumes evaluated on the raw-unit trace ' ...
                     '(60-s moving-mean baseline removed) between the same ' ...
                     'landmarks stored in bm; raw sensor units, not liters']);
    end
end

function [on, pk, tr, info] = applyOnsetRefinement(rspStored, det, fs, onIn, pkIn, trIn, refineStruct)
% Call refineInhaleOnsets_zlp on the detection's landmarks and CHECK what it
% returns (no repair: a violation is an error). Contract: onsets finite, inside
% the trace, strictly ascending after rounding; every onset has a peak after it;
% the first peak after onset k lies before onset k+1 (two onsets never share a
% peak). The refinement keeps peaks and troughs ([] returned = the input is
% kept) and returns the onsets one-to-one. info = the check record.
    onIn = double(onIn(:)'); pkIn = double(pkIn(:)'); trIn = double(trIn(:)');
    [on2, pk2, tr2, rinfo] = refineInhaleOnsets_zlp(rspStored, det, fs, onIn, pkIn, trIn, refineStruct);
    N = numel(rspStored);
    if isempty(pk2), pk2 = pkIn; end
    if isempty(tr2), tr2 = trIn; end
    on = round(double(on2(:)'));
    pk = sort(round(double(pk2(:)'))); tr = sort(round(double(tr2(:)')));
    bad = @(msg) error('zlpEstimateAllFeatures:inconsistentRefine', 'onset refinement result rejected: %s', msg);
    if any(~isfinite(on)) || any(on < 1) || any(on > N), bad('onsets must be finite and inside the trace'); end
    if any(~isfinite(pk)) || any(~isfinite(tr)), bad('peaks / troughs must be finite'); end
    if numel(on) ~= numel(onIn), bad(sprintf('%d onsets came back for %d', numel(on), numel(onIn))); end
    if any(diff(on) <= 0), bad('onsets must be strictly ascending after rounding (duplicates)'); end
    nOn = numel(on); pairPk = nan(1, nOn);
    for k = 1:nOn
        q = pk(pk > on(k));
        if isempty(q), bad(sprintf('onset %d (sample %d) has no peak after it', k, on(k))); end
        pairPk(k) = q(1);
        if k < nOn && pairPk(k) >= on(k + 1)
            bad(sprintf('onsets %d and %d (samples %d, %d) share the peak at sample %d', k, k + 1, on(k), on(k + 1), pairPk(k)));
        end
    end
    if ~isfield(rinfo, 'rule') || numel(rinfo.rule) ~= nOn || ~isfield(rinfo, 'shiftMs') || numel(rinfo.shiftMs) ~= nOn
        bad('the refinement record does not hold one rule and one shift per onset');
    end
    info = struct('onsetsIn', onIn, 'peaksIn', pkIn, 'troughsIn', trIn, 'onsetsOut', on, 'peaksOut', pk, 'troughsOut', tr, ...
        'nOnIn', numel(onIn), 'nOnOut', nOn, 'nPeaksIn', numel(pkIn), 'nPeaksOut', numel(pk), ...
        'nOrphanPeaksIn', numel(pkIn) - numel(unique(firstAfter(onIn, pkIn))), ...
        'nOrphanPeaksOut', numel(pk) - numel(unique(pairPk)), ...
        'rule', {reshape(rinfo.rule, 1, [])}, 'shiftMs', reshape(double(rinfo.shiftMs), 1, []), 'refineFn', rinfo);
end

function p = firstAfter(on, pk)
% the pairing used by the driver: for each onset the first peak after it
    p = nan(1, numel(on));
    for k = 1:numel(on)
        q = pk(pk > on(k));
        if ~isempty(q), p(k) = q(1); end
    end
    p = p(isfinite(p));
end
