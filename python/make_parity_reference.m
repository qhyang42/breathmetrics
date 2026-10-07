% make_parity_reference.m - export the MATLAB reference for the Python port
% parity check (python/parity_check.py).
%
% Run from the repository root (so sample_data.mat and the
% breathmetrics_functions are on the path):
%
%   >> addpath(genpath(pwd)); run('python/make_parity_reference.m')
%
% Writes, into python/parity_ref/:
%   resp.bin      the sample respiration trace, float64 little-endian
%   det.bin       the conditioned detection trace, float64
%   meta.csv      fs and array length
%   landmarks.csv prep peaks/troughs and the locked-method onsets + pruned
%                 extrema, ONE-BASED (MATLAB) sample indices, one row per
%                 landmark: kind,index

outDir = fullfile(fileparts(mfilename('fullpath')), 'parity_ref');
if ~exist(outDir, 'dir'), mkdir(outDir); end

S = load('sample_data.mat');
resp = double(S.resp(:))';
fs = double(S.srate);

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

fprintf('PARITYREF: fs=%g n=%d | prep %d pk / %d tr | final %d on / %d pk / %d tr\n', ...
    fs, numel(resp), numel(pk0), numel(tr0), numel(on), numel(pkP), numel(trP));
disp('PARITYREF DONE')
