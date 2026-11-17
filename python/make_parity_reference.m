% make_parity_reference.m - export the MATLAB reference for the Python port
% parity check (python/parity_check.py).
%
% Run from the repository root (so sample_data.mat and the
% breathmetrics_functions are on the path):
%
%   >> addpath(genpath(pwd)); run('python/make_parity_reference.m')
%
% Two traces are exported, each into its own folder of python/parity_ref/:
%   sample/   the toolbox's sample_data.mat
%   synth/    the deterministic synthetic trace of tests/zlpSyntheticTrace.m
%             (it contains double inhales, trough rebounds, lead bumps and
%             pause blips, so every merge step and every refinement rule fires)
% Each folder holds
%   resp.bin      the respiration trace, float64 little-endian
%   det.bin       the conditioned detection trace, float64
%   meta.csv      fs and array length
%   landmarks.csv prep peaks/troughs and the locked-method onsets + pruned
%                 extrema, ONE-BASED (MATLAB) sample indices, one row per
%                 landmark: kind,index
%   stages.mat    (-v7) the outputs of the stages after the detection, all
%                 ONE-BASED:  detection landmarks (engOn / engPk / engTr);
%                 refineInhaleOnsets_zlp on them with the default options
%                 (refOn, refInfo) and with three option variants (refVar);
%                 the driver's landmarks with refineOnsets false / true /
%                 true + mergeIncompleteExhales (driver*); the merge input
%                 (the driver's refined paired landmarks, mergeIn*) and the
%                 outputs of mergeIncompleteExhales_zlp on it with the
%                 defaults (merged, mergeInfo) and two option variants
%                 (mergeVar)

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'breathmetrics_functions'), fullfile(root, 'tests'));
set(0, 'DefaultFigureVisible', 'off');
outRoot = fullfile(root, 'python', 'parity_ref');

cases = struct('name', {'sample', 'synth'}, 'resp', {[], []}, 'fs', {[], []});
S = load(fullfile(root, 'sample_data.mat'));
cases(1).resp = double(S.resp(:))'; cases(1).fs = double(S.srate);
[xs, fss] = zlpSyntheticTrace();
cases(2).resp = double(xs(:))'; cases(2).fs = double(fss);

for c = 1:numel(cases)
    resp = cases(c).resp; fs = cases(c).fs;
    outDir = fullfile(outRoot, cases(c).name);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    [det, pk0, tr0] = prepBreathTrace_zlp(resp, fs, 'conservative', [], [], 0.05);
    [on, pkP, trP] = findInhaleOnsets_zlp(det, fs, pk0, tr0, 'kneeBacktrack', ...
        0.4, 1.25, 0.50, 0.10);

    fid = fopen(fullfile(outDir, 'resp.bin'), 'w');
    fwrite(fid, resp, 'float64', 'ieee-le'); fclose(fid);
    fid = fopen(fullfile(outDir, 'det.bin'), 'w');
    fwrite(fid, det, 'float64', 'ieee-le'); fclose(fid);

    fid = fopen(fullfile(outDir, 'meta.csv'), 'w');
    fprintf(fid, 'fs,%g\nn,%d\n', fs, numel(resp)); fclose(fid);

    fid = fopen(fullfile(outDir, 'landmarks.csv'), 'w');
    fprintf(fid, 'kind,index\n');
    w = @(kind, v) arrayfun(@(x) fprintf(fid, '%s,%d\n', kind, x), v);
    w('prepPeak', pk0); w('prepTrough', tr0);
    w('onset', on); w('peak', pkP); w('trough', trP);
    fclose(fid);

    % ---- onset refinement ----
    respFilled = fillmissing(resp, 'linear', 'EndValues', 'nearest');
    engOn = on; engPk = pkP; engTr = trP;
    [refOn, ~, ~, refInfo] = refineInhaleOnsets_zlp(respFilled, det, fs, on, pkP, trP);
    refVarOpts = {struct('levelRefMode', 'local'), struct('levelGuard', 'rise'), ...
        struct('levelGuard', 'none', 'levelTroughFrac', 0)};
    refVar = cell(1, numel(refVarOpts));
    for q = 1:numel(refVarOpts)
        [o2, ~, ~, i2] = refineInhaleOnsets_zlp(respFilled, det, fs, on, pkP, trP, refVarOpts{q});
        refVar{q} = struct('opts', refVarOpts{q}, 'on', o2, 'rule', {i2.rule}, 'shiftMs', i2.shiftMs, ...
            'confidence', i2.confidence, 'level', i2.level);
    end

    % ---- the driver: landmarks with the stages off / on ----
    bm0 = zlpEstimateAllFeatures(resp, fs, 'refineOnsets', false);
    [bm1, ~, z1] = zlpEstimateAllFeatures(resp, fs);
    [bm2, ~, z2] = zlpEstimateAllFeatures(resp, fs, 'mergeIncompleteExhales', true);
    driver0 = struct('on', bm0.inhaleOnsets, 'pk', bm0.inhalePeaks, 'tr', bm0.exhaleTroughs);
    driver1 = struct('on', bm1.inhaleOnsets, 'pk', bm1.inhalePeaks, 'tr', bm1.exhaleTroughs);
    driver2 = struct('on', bm2.inhaleOnsets, 'pk', bm2.inhalePeaks, 'tr', bm2.exhaleTroughs, ...
        'nTroughGuard', z2.merge.nTroughGuard);
    drvRefine = struct('onsetsOut', z1.refine.onsetsOut, 'rule', {z1.refine.rule}, 'shiftMs', z1.refine.shiftMs, ...
        'nOrphanPeaksIn', z1.refine.nOrphanPeaksIn, 'nOrphanPeaksOut', z1.refine.nOrphanPeaksOut);

    % ---- merge: input = the driver's refined paired landmarks ----
    n = numel(bm1.inhaleOnsets);
    mergeInOn = bm1.inhaleOnsets; mergeInPk = bm1.inhalePeaks(1:n-1); mergeInTr = bm1.exhaleTroughs(1:n-1);
    [merged, mergeInfo] = mergeIncompleteExhales_zlp(respFilled, fs, mergeInOn, mergeInPk, mergeInTr);
    mergeVarSpec = {struct('exFrac', 0.5, 'opts', struct()), ...
        struct('exFrac', 0.2, 'opts', struct('step3Mode', 'window', 'keepRule', 'first'))};
    mergeVar = cell(1, numel(mergeVarSpec));
    for q = 1:numel(mergeVarSpec)
        [m2, i2] = mergeIncompleteExhales_zlp(respFilled, fs, mergeInOn, mergeInPk, mergeInTr, ...
            mergeVarSpec{q}.exFrac, mergeVarSpec{q}.opts);
        mergeVar{q} = struct('spec', mergeVarSpec{q}, 'merged', m2, 'info', i2);
    end

    save(fullfile(outDir, 'stages.mat'), 'engOn', 'engPk', 'engTr', 'refOn', 'refInfo', 'refVar', ...
        'driver0', 'driver1', 'driver2', 'drvRefine', 'mergeInOn', 'mergeInPk', 'mergeInTr', ...
        'merged', 'mergeInfo', 'mergeVar', '-v7');

    fprintf('PARITYREF %s: fs=%g n=%d | prep %d pk / %d tr | final %d on / %d pk / %d tr | refined %d moved | merge %d -> %d\n', ...
        cases(c).name, fs, numel(resp), numel(pk0), numel(tr0), numel(on), numel(pkP), numel(trP), ...
        nnz(refOn ~= on), mergeInfo.nBefore, mergeInfo.nAfter);
end
disp('PARITYREF DONE')
