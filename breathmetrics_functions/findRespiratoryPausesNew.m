function [exhalepauseonsets, inhalepauseonsets] = ...
    findRespiratoryPausesNew(resp, srate, inhaleOnsets, ...
    exhaleTroughs, inhalePeaks, nBINS)
%%% find exhale offsets (start of between breath pauses) with line fitting

%%% need smoothed breathing signal, sampling rate, inhale onsets,
%%% exhale troughts

%%% what breathmetrics did for smoothing
% srate = 1000;
% smoothWinsize = 50;
% srateCorrectedSmoothedWindow = ...
%     floor((srate/1000) * smoothWinsize);
% smoothedRespiration = fftSmooth(resp, srateCorrectedSmoothedWindow)';


if nargin < 5
    inhalePeaks = [];
end
if nargin < 6 || isempty(nBINS)
    nBINS = floor(srate/100);
    if nBINS <= 20
        nBINS = 20;
    end
end

resp = resp(:);
inhaleOnsets = round(inhaleOnsets(:));
exhaleTroughs = round(exhaleTroughs(:));
inhalePeaks = round(inhalePeaks(:));

nTroughs = length(exhaleTroughs);
nOnsets = length(inhaleOnsets);
nPeaks = length(inhalePeaks);

% find exhale pause onsets (between trough and next inhale)
exhalepauseonsets = nan(nTroughs, 1);
for THISBREATH = 1:min(nTroughs, nOnsets - 1)
    thisStart = exhaleTroughs(THISBREATH);
    thisStop = inhaleOnsets(THISBREATH+1);
    if thisStart < 1 || thisStop > length(resp) || thisStop <= thisStart
        continue
    end
    inhaleWindow = resp(thisStart:thisStop);
    out = detect_pause_segment(inhaleWindow, srate);
    if ~isnan(out.tau_index)
        exhalepauseonsets(THISBREATH) = out.tau_index + thisStart - 1;
    end
end

% legacy inhale pause detection (between inhale peak and exhale onset)
inhalepauseonsets = nan(max(nPeaks, nTroughs), 1);
if nPeaks > 0 && nTroughs > 0
    if nBINS >= 100
        maxPauseBins = 5;
    else
        maxPauseBins = 2;
    end

    MAXIMUM_BIN_THRESHOLD = 5;
    UPPER_THRESHOLD = round(nBINS * .7);
    LOWER_THRESHOLD = round(nBINS * .3);

    nInhalePauseBreaths = min(nPeaks - 1, nTroughs);
    for THISBREATH = 1:max(nInhalePauseBreaths, 0)
        thisPeak = inhalePeaks(THISBREATH);
        thisTrough = exhaleTroughs(THISBREATH);
        if thisPeak < 1 || thisTrough > length(resp) || ...
                thisTrough <= thisPeak
            continue
        end

        exhaleWindow = resp(thisPeak:thisTrough);
        [isInhalePause, windowBins, modeBin, ampValues] = ...
            classifyPauseWindow(exhaleWindow, nBINS, LOWER_THRESHOLD, ...
            UPPER_THRESHOLD, MAXIMUM_BIN_THRESHOLD);
        if ~isInhalePause
            continue
        end

        [minPauseRange, maxPauseRange] = getPauseRange(windowBins, ...
            ampValues, modeBin, maxPauseBins);
        putativePauseInds = find(exhaleWindow > minPauseRange & ...
            exhaleWindow < maxPauseRange);
        if isempty(putativePauseInds)
            continue
        end

        pauseOnset = putativePauseInds(1) - 1;
        inhalepauseonsets(THISBREATH) = thisPeak + pauseOnset - 1;
    end
end
end


%% subfunctions
function [isPause, windowBins, modeBin, ampValues] = classifyPauseWindow( ...
    thisWindow, nBINS, lowerThreshold, upperThreshold, maxBinThreshold)
if isempty(thisWindow)
    windowBins = [];
    modeBin = 1;
    ampValues = 0;
    isPause = false;
    return
end

if max(thisWindow) == min(thisWindow)
    windowBins = linspace(min(thisWindow), max(thisWindow) + eps, nBINS);
else
    windowBins = linspace(min(thisWindow), max(thisWindow), nBINS);
end
[ampValues, windowBins] = hist(thisWindow, windowBins);
[~, modeBin] = max(ampValues);
if mean(ampValues) == 0
    maxBinRatio = 0;
else
    maxBinRatio = ampValues(modeBin) / mean(ampValues);
end
isPause = ~(modeBin < lowerThreshold || modeBin > upperThreshold || ...
    maxBinRatio < maxBinThreshold);
end

function [minPauseRange, maxPauseRange] = getPauseRange(windowBins, ...
    ampValues, modeBin, maxPauseBins)
if isempty(windowBins)
    minPauseRange = 0;
    maxPauseRange = 0;
    return
end

modeBin = min(max(modeBin, 1), length(windowBins));
nextBin = min(modeBin + 1, length(windowBins));
minPauseRange = windowBins(modeBin);
maxPauseRange = windowBins(nextBin);
maxBinTotal = ampValues(modeBin);
binningThreshold = .25;

for additionalBin = 1:maxPauseBins
    thisBin = modeBin - additionalBin;
    if thisBin < 1
        break
    end
    nValsAdded = ampValues(thisBin);
    if nValsAdded > maxBinTotal * binningThreshold
        minPauseRange = windowBins(thisBin);
    end
end

for additionalBin = 1:maxPauseBins
    thisBin = modeBin + additionalBin;
    if thisBin > length(windowBins)
        break
    end
    nValsAdded = ampValues(thisBin);
    if nValsAdded > maxBinTotal * binningThreshold
        maxPauseRange = windowBins(thisBin);
    end
end
end

function out = detect_pause_segment(y, fs, minEdgeMs, flatFrac)
% Detect a pause in one trough->next-inhale segment using BIC-penalized
% 1-step vs 2-step (continuous) piecewise-linear models.
%
% Required:
%   y  : column or row vector (segment)
%   fs : sampling rate in Hz
%
% Optional (positional):
%   minEdgeMs : ms of data to exclude at both ends when searching tau (default 200ms)
%   flatFrac    : |slope_early| <= flatFrac * |slope_late| (default 0.5)

if nargin < 3 || isempty(minEdgeMs), minEdgeMs = 200; end
if nargin < 4 || isempty(flatFrac),    flatFrac    = 0.5;  end

y = y(:);
n = numel(y);
if n < 3
    out = pack_output(n, fs, NaN, NaN, "one-step", NaN, NaN, NaN, NaN, ...
        false, []);
    return
end

t = (0:n-1)'/fs;
y0 = y - mean(y);

% ----- 1-step linear: y = a + b t
X1   = [ones(n,1), t];
b1   = X1 \ y0;
yhat1= X1 * b1;
rss1 = sum((y0 - yhat1).^2);
bic1 = n*log(rss1/n) + 2*log(n);

% ----- 2-step continuous hinge: y = a + b t + c * max(0, t - tau)
minEdge = max(1, floor(minEdgeMs/1000*fs));
if n - 2*minEdge < 1
    % Not enough samples to search a breakpoint; fall back to 1-step
    out = pack_output(n, fs, bic1, NaN, "one-step", NaN, NaN, b1(2), ...
        b1(2), false, yhat1);
    return
end

best_rss = inf;
best = struct();
for tauIdx = (minEdge:(n-minEdge))
    tau   = t(tauIdx);
    hinge = max(0, t - tau);
    X2    = [ones(n,1), t, hinge];
    b2    = X2 \ y0;
    yhat2 = X2 * b2;
    rss2  = sum((y0 - yhat2).^2);
    if rss2 < best_rss
        best_rss   = rss2;
        best.beta  = b2;             % [a; b; c]
        best.yhat  = yhat2;
        best.tau   = tau;
        best.tauIdx= tauIdx;
    end
end
bic2 = n*log(best_rss/n) + 4*log(n); % count tau as a parameter


% ----- Decision + pause rule
if bic2 < bic1
    b = best.beta(2); c = best.beta(3);
    slopeEarly = b;
    slopeLate  = b + c;
    if best.beta(3)<0 && (abs(slopeEarly) * flatFrac >= slopeLate)% slope 2 needs to be smaller than slope 1
        %         if best.beta(3)<0 && (early_ms >= minPauseMs)

        chosen = "two-step";
        pauseDetected = true;
        out = pack_output(n, fs, bic1, bic2, chosen, best.tauIdx, best.tau, ...
            slopeEarly, slopeLate, pauseDetected, best.yhat);
    else
        chosen = "one-step";
        out = pack_output(n, fs, bic1, bic2, chosen, NaN, NaN, ...
            b1(2), b1(2), false, yhat1);
    end
else
    chosen = "one-step";
    out = pack_output(n, fs, bic1, bic2, chosen, NaN, NaN, ...
        b1(2), b1(2), false, yhat1);
end
end


function out = pack_output(n, fs, bic1, bic2, chosen, tauIdx, tauSec, s1, s2, pauseDetected, yhat)
out = struct( ...
    'n_samples',    n, ...
    'duration_sec', n/fs, ...
    'bic_one_step', bic1, ...
    'bic_two_step', bic2, ...
    'chosen_model', chosen, ...
    'tau_index',    tauIdx, ...
    'tau_sec',      tauSec, ...
    'slope_early',  s1, ...
    'slope_late',   s2, ...
    'pause_detected', logical(pauseDetected), ...
    'yhat',         yhat );
end
