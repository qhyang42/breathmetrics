function [inhaleOnsets, dbg, exhaleOnsets] = findRespiratoryOnsetsNew( ...
    resp, fs, peaks, troughs, nBINS)
% detectBreathOnsetsFromPeaks
%
% Stand-alone inhale-onset detection extracted from breathTemplates4.m
% (logic preserved: 3s lookback -> findInflection -> pass2 refine via
% findInflectionFromMid3 -> final bStart).
%
% INPUTS
%   resp  : breathing signal (vector)
%   fs    : sampling rate (Hz)
%   peaks : respiratory peak indices (samples)
%   troughs (optional): exhale-trough indices (samples)
%   nBINS (optional): legacy histogram bins for exhale onset estimation
%
% OUTPUTS
%   inhaleOnsets : inhale-onset index for each peak (samples)
%   dbg          : struct with intermediate values (adj, adj2, etc.)
%   exhaleOnsets : exhale-onset index for each trough (samples), estimated
%                  with legacy BreathMetrics logic when troughs are passed.
%
% NOTES
% - This includes the original helper code as local subfunctions.
% - The original collaborator code assumes peaks are far enough from the
%   signal boundaries (>= 3s from start, and >= fs/5 from end). Here we
%   lightly clamp windows to avoid indexing errors, but the algorithmic
%   steps are otherwise identical.

resp  = resp(:);
peaks = peaks(:);
nSamp = numel(resp);
exhaleOnsets = [];

if nargin < 4
    troughs = [];
end
if nargin < 5 || isempty(nBINS)
    nBINS = floor(fs/100);
    if nBINS <= 20
        nBINS = 20;
    end
end

inhaleOnsets = nan(size(peaks));

dbg = struct();
dbg.winStart  = nan(size(peaks));
dbg.winEnd1   = nan(size(peaks));
dbg.winEnd2   = nan(size(peaks));
dbg.adj       = nan(size(peaks));
dbg.adj2      = nan(size(peaks));
dbg.bStartTmp = nan(size(peaks));

for ii = 1:numel(peaks)
    curidx = peaks(ii);
    % ---- PASS 1 (large window; heavy smoothing) ----
    winStart = curidx - fs*3;
    winEnd1  = curidx;

    % clamp to signal bounds (to prevent crashes at edges)
    winStart = max(1, winStart);
    winEnd1  = min(nSamp, winEnd1);

    inSig = smoothdata(resp(winStart:winEnd1), 'gaussian', round(fs));

    % find initial guess
    [adj, ~] = findInflection(inSig, true, false, []);


    % ---- PASS 2 (slightly larger window; moderate smoothing; refine) ----
    % winEnd2 = curidx + round(fs/5);
    winEnd2 = curidx; % QY 1/12/2026 I like this better than large window because large window makes inhale onset a little early. could be data specific.
    winEnd2 = min(nSamp, winEnd2);

    inSig2 = resp(winStart:winEnd2);
    inSigRaw = inSig2;
    inSig2 = smoothdata(inSig2, 'gaussian', round(fs/2));

    % second guess
    adj2 = findInflectionFromMid3(inSig2, adj, 0, fs, inSigRaw);

    % result
    bStartTmp = adj2+winStart;
    bStart = bStartTmp;
    inhaleOnsets(ii) = bStart;

    % debug
    dbg.winStart(ii)  = winStart;
    dbg.winEnd1(ii)   = winEnd1;
    dbg.winEnd2(ii)   = winEnd2;
    dbg.adj(ii)       = adj;
    dbg.adj2(ii)      = adj2;
    dbg.bStartTmp(ii) = bStartTmp;
end

if ~isempty(troughs)
    exhaleOnsets = detectExhaleOnsetsLegacy(resp, peaks, troughs, ...
        nBINS, fs);
end
end

function exhaleOnsets = detectExhaleOnsetsLegacy(resp, peaks, troughs, ...
    nBINS, fs)
% Legacy exhale-onset detection from findRespiratoryPausesAndOnsets.m

resp = resp(:)';
peaks = round(peaks(:)');
troughs = round(troughs(:)');
nPeaks = length(peaks);
nTroughs = length(troughs);
exhaleOnsets = nan(1, nTroughs);

if isempty(resp) || nPeaks == 0 || nTroughs == 0
    return
end

if nBINS >= 100
    maxPauseBins = 5;
else
    maxPauseBins = 2;
end

MAXIMUM_BIN_THRESHOLD = 5;
UPPER_THRESHOLD = round(nBINS * .7);
LOWER_THRESHOLD = round(nBINS * .3);
SIMPLE_ZERO_CROSS = mean(resp);

nLoop = min(max(nPeaks - 1, 0), nTroughs);
for thisBreath = 1:nLoop
    thisPeak = peaks(thisBreath);
    thisTrough = troughs(thisBreath);
    if thisTrough <= thisPeak
        exhaleOnsets(thisBreath) = thisPeak;
        continue
    end

    exhaleWindow = resp(thisPeak:thisTrough);
    [isInhalePause, windowBins, modeBin, ampValues] = ...
        classifyPauseWindow(exhaleWindow, nBINS, LOWER_THRESHOLD, ...
        UPPER_THRESHOLD, MAXIMUM_BIN_THRESHOLD);

    if ~isInhalePause
        possibleExhaleInds = exhaleWindow > SIMPLE_ZERO_CROSS;
        exhaleRel = find(possibleExhaleInds == 1, 1, 'last');
        if isempty(exhaleRel)
            exhaleOnsets(thisBreath) = thisPeak;
        else
            exhaleOnsets(thisBreath) = thisPeak + exhaleRel - 1;
        end
    else
        [minPauseRange, maxPauseRange] = getPauseRange(windowBins, ...
            ampValues, modeBin, maxPauseBins);
        putativePauseInds = find(exhaleWindow > minPauseRange & ...
            exhaleWindow < maxPauseRange);
        if isempty(putativePauseInds)
            possibleExhaleInds = exhaleWindow > SIMPLE_ZERO_CROSS;
            exhaleRel = find(possibleExhaleInds == 1, 1, 'last');
            if isempty(exhaleRel)
                exhaleOnsets(thisBreath) = thisPeak;
            else
                exhaleOnsets(thisBreath) = thisPeak + exhaleRel - 1;
            end
        else
            exhaleRel = putativePauseInds(end) + 1;
            exhaleOnsets(thisBreath) = min(thisPeak + exhaleRel - 1, ...
                thisTrough);
        end
    end
end

lastPair = min(nPeaks, nTroughs);
if lastPair < 1
    return
end

if nPeaks > 1
    tailOnsetLims = floor(mean(diff(peaks)));
else
    tailOnsetLims = floor(fs);
end
if ~isfinite(tailOnsetLims) || tailOnsetLims < 1
    tailOnsetLims = floor(fs);
end

if length(resp) - peaks(lastPair) > tailOnsetLims
    lastBoundary = peaks(lastPair) + tailOnsetLims;
else
    lastBoundary = length(resp);
end
lastBoundary = min(max(lastBoundary, peaks(lastPair)), length(resp));

exhaleWindow = resp(peaks(lastPair):lastBoundary);
possibleExhaleInds = exhaleWindow < SIMPLE_ZERO_CROSS;
if sum(possibleExhaleInds) > 0
    exhaleBestGuess = find(possibleExhaleInds == 1, 1, 'first');
    exhaleOnsets(lastPair) = peaks(lastPair) + exhaleBestGuess - 1;
else
    exhaleOnsets(lastPair) = lastBoundary;
end
end

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


% ========================================================================
% Helper functions (verbatim from collaborator code, as local subfunctions)
% ========================================================================

function [inflection, SSDer] = findInflection(sig, slopeBased, ...
    plotIT, tim)

%Sig      1Xtime timeseries vector

%inflection   integer index value specifying the nearest major inflection
%point in the data from the end

% sig = normalize(sig);

[SS, slopes] = suffixLineFitStats(sig);

%calculate SS derivative:
SSDer = diff(SS);
SSDer = [SSDer(1); SSDer]; %pad

smoothSpan = max(1, round(length(sig)/20));
slopes = smoothdata(slopes, 'gaussian', smoothSpan);

if slopeBased

    slopes = flipud(slopes);
    [~, maxidx] = max(slopes);

    inflection = maxidx;

else
    SSDer = flipud(SSDer);
    [~, maxidx] = max(SSDer);
    inflection = maxidx;
end

if plotIT

    if isempty(tim)
        tim = 1:length(sig);
    end

    figure;
    subplot 211
    hold on
    plot(tim, sig, 'linewidth', 2)
    scatter(tim(end - inflection + 1), sig(end - inflection + 1), ...
        80, 'r', 'filled')
    title('signal and inflection')

    subplot 212
    if slopeBased
        plot(flipud(slopes), 'linewidth', 2)
        title('slopes (flipped)')
    else
        plot(flipud(SSDer), 'linewidth', 2)
        title('SS derivative (flipped)')
    end

end

end


function [inflection] = findInflectionFromMid3(sig, midIDX, plotPos, fs, ...
    sigRaw)

%Sig      1Xtime timeseries vector
%midIDX   index of candidate inflection from the END

%inflection   integer index value specifying the nearest major inflectio
%point in the data from the end

%plotPos  determine plot positioning. 1 indicates 1st plot, 2 indicates
%second plot.

if nargin < 3
    plotPos = 0;
end

if nargin < 4
    fs = 1000;
end

N = length(sig);

%Start at the initial inflection value
starti = N - midIDX + 1 - round(fs/5); %fwd shift
endi = N - 5;

if starti < 1
    starti = 1;
end

if endi > N
    endi = N;
end

if plotPos == 0
    plotIT = false;
else
    plotIT = true;
end
%find inflection upward with smaller window
inflection = findInflectionUpward(sig(starti:endi), plotIT, plotPos, ...
    sigRaw(starti:endi));

%adjust inflection to correspond to original signal indices
inflection = inflection + starti - 1;

end


function [inflection] = findInflectionUpward(sig, plotIT, plotPos, sigRaw)

if nargin < 2
    plotIT = false;
end

if nargin < 3
    plotPos = 0;
end

if nargin < 4
    sigRaw = sig;
end

N = length(sig);

[~, slopes] = suffixLineFitStats(sig);

smoothSpan = max(1, round(length(sig)/20));
slopes = smoothdata(slopes, 'gaussian', smoothSpan);
slopes = flipud(slopes);

% NOTE: monoCheck is used to enforce that we pick an inflection that is
% monotonic (in the raw data) in the neighborhood around the candidate.
% Evaluate the mask once so candidate selection stays linear-time.
monoMask = flipud(monoCheckMask(sigRaw));
candidateSlopes = slopes;
candidateSlopes(~monoMask) = -inf;

[bestSlope, bestCand] = max(candidateSlopes);
if isfinite(bestSlope)
    inflection = N - bestCand + 1;
else
    [~, bestCand] = max(slopes);
    inflection = N - bestCand + 1;
end

if plotIT
    figure;
    if plotPos == 1
        subplot 211
    elseif plotPos == 2
        subplot 212
    end
    hold on
    plot(sig, 'linewidth', 2)
    scatter(N - inflection + 1, sig(N - inflection + 1), 80, 'r', 'filled')
    title('findInflectionUpward: signal & chosen inflection')
end

end


function [rss, slopes] = suffixLineFitStats(sig)
sig = sig(:);
N = length(sig);

rss = zeros(N, 1);
slopes = zeros(N, 1);
if N < 2
    return
end

x = (1:N)';
suffixCount = (N:-1:1)';
suffixX = flipud(cumsum(flipud(x)));
suffixX2 = flipud(cumsum(flipud(x.^2)));
suffixY = flipud(cumsum(flipud(sig)));
suffixY2 = flipud(cumsum(flipud(sig.^2)));
suffixXY = flipud(cumsum(flipud(x .* sig)));

den = suffixCount .* suffixX2 - suffixX.^2;
valid = den ~= 0;
slopes(valid) = (suffixCount(valid) .* suffixXY(valid) - ...
    suffixX(valid) .* suffixY(valid)) ./ den(valid);

intercepts = zeros(N, 1);
intercepts(valid) = (suffixY(valid) - slopes(valid) .* suffixX(valid)) ./ ...
    suffixCount(valid);

rss(valid) = suffixY2(valid) - ...
    (intercepts(valid) .* suffixY(valid) + slopes(valid) .* ...
    suffixXY(valid));
rss = max(rss, 0);
end


function monoMask = monoCheckMask(vecIn2)
vecIn2 = vecIn2(:);
N = length(vecIn2);
monoMask = false(N, 1);
if N < 11
    return
end

for peakidx = 6:(N - 5)
    preVal = min(vecIn2(peakidx-5:peakidx-1));
    postVal = max(vecIn2(peakidx+1:peakidx+5));
    monoMask(peakidx) = postVal > preVal;
end
end


function [isMono] = monoCheck(vecIn2, peakidx)

% monoCheck: returns true if the signal is monotonic (increasing) around
% peakidx within a small neighborhood; false otherwise.

% default to false
isMono = false;

% guard
if peakidx - 5 < 1 || peakidx + 5 > length(vecIn2)
    return
end

% check monotonicity-ish: local rise must be sufficiently large
preVal  = min(vecIn2(peakidx-5:peakidx-1));
postVal = max(vecIn2(peakidx+1:peakidx+5));

if postVal > preVal
    isMono = true;
end

end
