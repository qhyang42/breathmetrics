function shapeFeatures = calculateRespiratoryShapeFeatures(resp, srate, ...
    inhaleOnsets, inhalePeaks, exhaleTroughs, exhaleOnsets, exhaleOffsets)
%CALCULATERESPIRATORYSHAPEFEATURES Calculate aligned per-breath shape features.
% The output columns and segment definitions mirror breathmetrics-py:
% inhale onset->peak, inhale peak->exhale trough, exhale onset->offset,
% and inhale onset->exhale offset. Segment endpoints are inclusive.

resp = resp(:);
inhaleOnsets = inhaleOnsets(:);
nBreaths = numel(inhaleOnsets);

breath_id = int64((0:(nBreaths - 1))');
missingValues = nan(nBreaths, 1);
shapeFeatures = table( ...
    breath_id, ...
    missingValues, missingValues, missingValues, ...
    missingValues, missingValues, missingValues, ...
    missingValues, missingValues, ...
    'VariableNames', { ...
        'breath_id', ...
        'inhale_smoothness', ...
        'inpeak_to_extrough_smoothness', ...
        'exhale_smoothness', ...
        'inhale_curvature', ...
        'inpeak_to_extrough_curvature', ...
        'exhale_curvature', ...
        'inpeak_to_extrough_slope', ...
        'max_flowchange'});

for breathIdx = 1:nBreaths
    inhale = getInclusiveSegment(resp, inhaleOnsets, inhalePeaks, ...
        breathIdx);
    inpeakToExtrough = getInclusiveSegment(resp, inhalePeaks, ...
        exhaleTroughs, breathIdx);
    exhale = getInclusiveSegment(resp, exhaleOnsets, exhaleOffsets, ...
        breathIdx);
    fullBreath = getInclusiveSegment(resp, inhaleOnsets, ...
        exhaleOffsets, breathIdx);

    if ~isempty(inhale)
        features = safeGetSegmentFeatures(inhale, srate);
        shapeFeatures.inhale_smoothness(breathIdx) = features.smoothness;
        shapeFeatures.inhale_curvature(breathIdx) = features.curvature;
    end

    if ~isempty(inpeakToExtrough)
        features = safeGetSegmentFeatures(inpeakToExtrough, srate);
        shapeFeatures.inpeak_to_extrough_smoothness(breathIdx) = ...
            features.smoothness;
        shapeFeatures.inpeak_to_extrough_curvature(breathIdx) = ...
            features.curvature;
        shapeFeatures.inpeak_to_extrough_slope(breathIdx) = ...
            features.phase2slope;
    end

    if ~isempty(exhale)
        features = safeGetSegmentFeatures(exhale, srate);
        shapeFeatures.exhale_smoothness(breathIdx) = features.smoothness;
        shapeFeatures.exhale_curvature(breathIdx) = features.curvature;
    end

    if ~isempty(fullBreath)
        features = safeGetSegmentFeatures(fullBreath, srate);
        shapeFeatures.max_flowchange(breathIdx) = features.maxFlowchange;
    end
end
end


function segment = getInclusiveSegment(signal, startEvents, endEvents, ...
    breathIdx)
startIdx = getEventAt(startEvents, breathIdx, numel(signal));
endIdx = getEventAt(endEvents, breathIdx, numel(signal));

if isnan(startIdx) || isnan(endIdx) || startIdx > endIdx
    segment = [];
    return
end

segment = signal(startIdx:endIdx);
if isempty(segment) || any(~isfinite(segment))
    segment = [];
end
end


function eventIdx = getEventAt(events, breathIdx, signalSize)
eventIdx = NaN;
if isempty(events) || breathIdx > numel(events)
    return
end

value = double(events(breathIdx));
if ~isscalar(value) || ~isfinite(value) || value ~= fix(value) || ...
        value < 1 || value > signalSize
    return
end
eventIdx = value;
end


function features = safeGetSegmentFeatures(segment, srate)
features = struct( ...
    'smoothness', NaN, ...
    'curvature', NaN, ...
    'phase2slope', NaN, ...
    'maxFlowchange', NaN);
try
    calculated = getSegmentFeatures(segment, srate);
    featureNames = fieldnames(features);
    for featureIdx = 1:numel(featureNames)
        featureName = featureNames{featureIdx};
        value = calculated.(featureName);
        if isscalar(value) && isfinite(value)
            features.(featureName) = double(value);
        end
    end
catch
    % Invalid or numerically degenerate segments remain NaN, matching Python.
end
end
