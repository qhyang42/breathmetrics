function segmentFeatures = getSegmentFeatures(signal, fs, pointIdx)
%GETSEGMENTFEATURES Segment-level features from Sagar et al. 2025.
% These features are calculated for breathing phases, not the whole signal.

if nargin < 3
    pointIdx = [];
end

signal = signal(:);
fs = double(fs);

segmentFeatures = struct( ...
    'smoothness', calculateSmoothness(signal, fs), ...
    'curvature', calculateCurvature(signal), ...
    'phase2slope', calculatePhase2Slope(signal), ...
    'maxFlowchange', calculateMaxFlowchange(signal), ...
    'timeSymmetry', computeTimeSymmetry(signal), ...
    'smoothnessAroundPoint', NaN, ...
    'timeSymmetryAroundPoint', NaN);

if ~isempty(pointIdx) && isfinite(pointIdx)
    pointIdx = round(pointIdx);
    pointIdx = min(max(pointIdx, 1), numel(signal));
    segmentFeatures.smoothnessAroundPoint = ...
        calculateSmoothnessAroundPoint(signal, fs, pointIdx);
    segmentFeatures.timeSymmetryAroundPoint = ...
        computeTimeSymmetryAroundPoint(signal, pointIdx);
end
end


function smoothness = calculateSmoothness(signal, fs)
% Compute smoothness as the negative mean absolute deviation from a moving average.
if isempty(signal)
    smoothness = NaN;
    return
end

window = max(1, round(fs / 10));
kernel = ones(window, 1) / window;
smoothSig = conv(signal, kernel, 'same');
smoothness = -mean(abs(smoothSig - signal));
end


function curvature = calculateCurvature(signal)
% Compute curvature as the mean absolute second derivative.
if numel(signal) < 2
    curvature = NaN;
    return
end

firstDerivative = gradient(signal);
secondDerivative = gradient(firstDerivative);
curvature = mean(abs(secondDerivative));
end


function maxFlowchange = calculateMaxFlowchange(signal)
% Compute the maximum consecutive-sample increase in the segment.
if numel(signal) < 2
    maxFlowchange = NaN;
    return
end

maxFlowchange = max(diff(signal));
end


function slope = calculatePhase2Slope(signal)
% Compute the slope of the breathing segment.
if numel(signal) < 2
    slope = NaN;
    return
end

x = (0:(numel(signal) - 1))';
p = polyfit(x, signal, 1);
slope = p(1);
end


function smoothness = calculateSmoothnessAroundPoint(signal, fs, pointIdx)
% Compute smoothness around a specific point in the segment.
segment = getPointWindow(signal, pointIdx);
smoothness = calculateSmoothness(segment, fs);
end


function corrVal = computeTimeSymmetry(signal)
% Compute time symmetry as the correlation with the time-reversed signal.
if numel(signal) < 2
    corrVal = NaN;
    return
end

flippedSignal = flipud(signal);
[signal, flippedSignal] = removeMiddleSample(signal, flippedSignal);

if std(signal) > 0 && std(flippedSignal) > 0
    corrMat = corrcoef(signal, flippedSignal);
    corrVal = corrMat(1, 2);
else
    corrVal = NaN;
end
end


function corrVal = computeTimeSymmetryAroundPoint(signal, pointIdx)
% Compute time symmetry in a local window around a point.
segment = getPointWindow(signal, pointIdx);
corrVal = computeTimeSymmetry(segment);
end


function segment = getPointWindow(signal, pointIdx)
windowSize = 0.05 * numel(signal);
halfWindowSamples = round(windowSize / 2);
startIdx = max(1, pointIdx - halfWindowSamples);
endIdx = min(numel(signal), pointIdx + halfWindowSamples - 1);
segment = signal(startIdx:endIdx);
end


function [signalOut, flippedOut] = removeMiddleSample(signalIn, flippedIn)
signalOut = signalIn;
flippedOut = flippedIn;

if mod(numel(signalIn), 2) ~= 0
    midIdx = floor(numel(signalIn) / 2) + 1;
    signalOut(midIdx) = [];
    flippedOut(midIdx) = [];
end
end
