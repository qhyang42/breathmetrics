function tests = testZlpRefineAndMerge
tests = functiontests(localfunctions);
end


function setupOnce(testCase)
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'breathmetrics_functions'));
addpath(fullfile(repoRoot, 'tests'));
set(0, 'DefaultFigureVisible', 'off');

d = load(fullfile(repoRoot, 'sample_data.mat'));
testCase.TestData.sample.resp = d.resp;
testCase.TestData.sample.srate = d.srate;
[x, fs, info] = zlpSyntheticTrace();
testCase.TestData.synth.resp = x;
testCase.TestData.synth.srate = fs;
testCase.TestData.synth.info = info;

names = {'sample', 'synth'};
for q = 1:numel(names)
    S = testCase.TestData.(names{q});
    [S.bmRefined, ~, S.zRefined] = zlpEstimateAllFeatures(S.resp, S.srate);
    [S.bmPlain, ~, S.zPlain] = zlpEstimateAllFeatures(S.resp, S.srate, 'refineOnsets', false);
    [S.bmMerged, ~, S.zMerged] = zlpEstimateAllFeatures(S.resp, S.srate, 'mergeIncompleteExhales', true);
    testCase.TestData.(names{q}) = S;
end
end


function testRefineOffReproducesDetection(testCase)
% 'refineOnsets', false: the landmarks are exactly those of the detection (the earlier behaviour of the driver)
names = {'sample', 'synth'};
for q = 1:numel(names)
    S = testCase.TestData.(names{q});
    [on, pk, tr] = detectLandmarks(S.resp, S.srate);
    verifyEqual(testCase, S.bmPlain.inhaleOnsets, on);
    verifyEqual(testCase, S.bmPlain.inhalePeaks, pk);
    verifyEqual(testCase, S.bmPlain.exhaleTroughs, tr);
    verifyEmpty(testCase, S.zPlain.refine);
    verifyEmpty(testCase, S.zPlain.merge);
end
end


function testRefinedOnsetsAreConsistent(testCase)
names = {'sample', 'synth'};
for q = 1:numel(names)
    S = testCase.TestData.(names{q});
    r = S.zRefined.refine;
    on = r.onsetsOut; pk = r.peaksOut;
    verifyTrue(testCase, all(diff(on) > 0), 'onsets strictly ascending');
    verifyEqual(testCase, r.nOnOut, r.nOnIn, 'one refined onset per detected onset');
    verifyEqual(testCase, r.peaksOut, r.peaksIn, 'peaks unchanged');
    verifyEqual(testCase, r.troughsOut, r.troughsIn, 'troughs unchanged');
    for k = 1:numel(on)
        nextPk = pk(find(pk > on(k), 1));
        verifyNotEmpty(testCase, nextPk);
        prevPk = pk(find(pk < on(k), 1, 'last'));
        if ~isempty(prevPk), verifyGreaterThan(testCase, on(k), prevPk); end
        if k < numel(on), verifyLessThan(testCase, nextPk, on(k + 1), 'two onsets share a peak'); end
    end
    verifyEqual(testCase, numel(S.bmRefined.inhaleOnsets), numel(S.bmPlain.inhaleOnsets), ...
        'refinement adds / drops no breath');
    verifyEqual(testCase, S.bmRefined.inhalePeaks, S.bmPlain.inhalePeaks);
    verifyEqual(testCase, S.bmRefined.exhaleTroughs, S.bmPlain.exhaleTroughs);
    verifyEqual(testCase, r.shiftMs, (r.onsetsOut - r.onsetsIn) / S.srate * 1000, 'AbsTol', 1e-9);
    verifyEqual(testCase, cellfun(@isempty, r.rule), r.shiftMs == 0);
    verifyGreaterThan(testCase, nnz(r.shiftMs ~= 0), 0, 'the refinement moves onsets');
end
end


function testSyntheticTraceTriggersEveryRule(testCase)
S = testCase.TestData.synth;
ct = S.zRefined.refine.refineFn.counts;
verifyGreaterThan(testCase, ct.nFoot, 0, 'rule foot');
verifyGreaterThan(testCase, ct.nFar, 0, 'rule far');
verifyGreaterThan(testCase, ct.nLevel, 0, 'rule level');
mi = S.zMerged.merge.info;
verifyGreaterThan(testCase, mi.nLeadBumps, 0, 'merge step 2: lead bumps');
verifyGreaterThan(testCase, sum(mi.isMinor & ~mi.isLeadBump), 0, 'merge step 1: pause blips');
verifyGreaterThan(testCase, mi.nMerged, 0, 'merge step 3');
verifyGreaterThan(testCase, mi.nTroughHeads, 0, 'merge keep rule');
end


function testMergeJoinsDoubleInhales(testCase)
S = testCase.TestData.synth;
z = S.zMerged.merge;
mg = z.merged; mi = z.info;
verifyEqual(testCase, mi.nBefore - mi.nAfter, mi.nMinor + mi.nMerged, 'record counts add up');
verifyEqual(testCase, numel(mg.onsets), mi.nAfter);
verifyEqual(testCase, sum(mg.nRows), mi.nBefore);
ev = [S.info.events(strcmp({S.info.events.type}, 'double')).tStart];
% every double-inhale pattern ends up as ONE breath that holds two inhale pulses
for k = 1:numel(ev)
    t0 = ev(k);
    q = find(mg.onsets / S.srate > t0 - 0.5 & mg.onsets / S.srate < t0 + 3.5 & mg.nPulses >= 2, 1);
    verifyNotEmpty(testCase, q, sprintf('double inhale at %.1f s joined', t0));
end
end


function testMergedObjectIsConsistent(testCase)
names = {'sample', 'synth'};
for q = 1:numel(names)
    S = testCase.TestData.(names{q});
    bm = S.bmMerged;
    mi = S.zMerged.merge.info;
    on = bm.inhaleOnsets; pk = bm.inhalePeaks; tr = bm.exhaleTroughs;
    verifyEqual(testCase, numel(on), mi.nAfter + 1, 'one landmark set per merged breath plus the closing onset');
    verifyTrue(testCase, all(on(1:end - 1) < pk(1:end - 1) & pk(1:end - 1) < tr(1:end - 1) & tr(1:end - 1) < on(2:end)), ...
        'onset < peak < trough < next onset');
    verifyLessThanOrEqual(testCase, numel(on), numel(S.bmRefined.inhaleOnsets));
    % the merged landmarks are the ones the object was built from
    mg = S.zMerged.merge.merged;
    verifyEqual(testCase, on(1:end - 1), mg.onsets);
    verifyEqual(testCase, pk(1:end - 1), mg.peaks);
    verifyEqual(testCase, tr(1:end - 1), mg.troughs);
    % the features were recomputed from them: one duration / volume per merged breath, all finite where defined
    verifyEqual(testCase, numel(bm.inhaleDurations), numel(on));
    verifyTrue(testCase, all(bm.inhaleDurations(1:end - 1) > 0));
end
end


function testMergeFunctionDirect(testCase)
% the landmark-level function, called on the refined landmarks of the synthetic trace
S = testCase.TestData.synth;
r = S.zRefined.refine;
on = r.onsetsOut; pk = r.peaksOut; tr = r.troughsOut;
n = numel(on);
pkA = nan(1, n); trA = nan(1, n);
for k = 1:n
    pk1 = pk(find(pk > on(k), 1)); pkA(k) = pk1;
    t1 = tr(find(tr > pk1, 1)); if ~isempty(t1), trA(k) = t1; end
end
while ~isfinite(trA(end)), on(end) = []; pkA(end) = []; trA(end) = []; end
[m1, i1] = mergeIncompleteExhales_zlp(S.resp, S.srate, on, pkA, trA);
[m2, i2] = mergeIncompleteExhales_zlp(S.resp, S.srate, on, pkA(1:end - 1), trA(1:end - 1), 0.20, struct());
verifyEqual(testCase, m1, m2, 'a 1 x m peak / trough array and a 1 x (m-1) one give the same result');
verifyEqual(testCase, i1.nAfter, numel(m1.onsets));
verifyEqual(testCase, m1.keptRow, cellfun(@(c) c(1), m1.origRows));
verifyTrue(testCase, all(diff(m1.onsets) > 0));
verifyEqual(testCase, m1.ends(1:end - 1), m1.onsets(2:end), 'each breath ends where the next starts');
verifyEqual(testCase, m1.ends(end), on(end));
% a stricter exhale fraction cannot join more inhales than the default
[~, i3] = mergeIncompleteExhales_zlp(S.resp, S.srate, on, pkA, trA, 0.05);
verifyLessThanOrEqual(testCase, i3.nMerged, i1.nMerged);
% argument checks
verifyError(testCase, @() mergeIncompleteExhales_zlp(S.resp, S.srate, on, pkA(1:end - 3), trA), ...
    'mergeIncompleteExhales_zlp:landmarks');
verifyError(testCase, @() mergeIncompleteExhales_zlp(S.resp, S.srate, fliplr(on), pkA, trA), ...
    'mergeIncompleteExhales_zlp:onsets');
end


function testRefineOptionsAreChecked(testCase)
S = testCase.TestData.synth;
verifyError(testCase, @() zlpEstimateAllFeatures(S.resp(1:60 * S.srate), S.srate, 'refineOpts', struct('noSuchOption', 1)), ...
    'refineInhaleOnsets_zlp:unknownOption');
end


function [on, pk, tr] = detectLandmarks(resp, srate)
% the detection and pairing of zlpEstimateAllFeatures without the refinement and the merge
respFilled = fillmissing(double(resp(:)'), 'linear', 'EndValues', 'nearest');
[det, pk0, tr0] = prepBreathTrace_zlp(respFilled, srate, 'conservative', [], [], 0.05);
[on, pkP, trP] = findInhaleOnsets_zlp(det, srate, pk0, tr0, 'kneeBacktrack', 0.4, 1.25, 0.50, 0.10);
on = round(sort(on(:)'));
n0 = numel(on); pkA = nan(1, n0); trA = nan(1, n0);
for k = 1:n0
    p = pkP(pkP > on(k)); if isempty(p), continue; end
    pkA(k) = p(1);
    t = trP(trP > pkA(k)); if ~isempty(t), trA(k) = t(1); end
end
keep = isfinite(pkA);
on = on(keep); pkA = pkA(keep); trA = trA(keep);
while ~isempty(on) && ~isfinite(trA(end)), on(end) = []; pkA(end) = []; trA(end) = []; end
pk = round(pkA); tr = round(trA);
end
