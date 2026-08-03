function tests = testShapeFeatures
tests = functiontests(localfunctions);
end


function setupOnce(testCase)
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'breathmetrics_functions'));
testCase.TestData.repoRoot = repoRoot;
end


function testShapeTableMatchesExpectedSegments(testCase)
signal = [ ...
    0, 0, 0.1, 0.4, 0.7, 1.1, 1.4, 1.0, 0.6, 0.1, ...
    -0.4, -0.9, -1.2, -1.0, -0.8, -0.5, -0.3, -0.1, 0, ...
    0.05, 0, 0];

shapeFeatures = calculateRespiratoryShapeFeatures( ...
    signal, 20, 3, 7, 13, 14, 21);

expectedNames = { ...
    'breath_id', ...
    'inhale_smoothness', ...
    'inpeak_to_extrough_smoothness', ...
    'exhale_smoothness', ...
    'inhale_curvature', ...
    'inpeak_to_extrough_curvature', ...
    'exhale_curvature', ...
    'inpeak_to_extrough_slope', ...
    'max_flowchange'};
verifyEqual(testCase, shapeFeatures.Properties.VariableNames, expectedNames);
verifyEqual(testCase, height(shapeFeatures), 1);
verifyEqual(testCase, shapeFeatures.breath_id, int64(0));

inhaleFeatures = getSegmentFeatures(signal(3:7), 20);
transitionFeatures = getSegmentFeatures(signal(7:13), 20);
exhaleFeatures = getSegmentFeatures(signal(14:21), 20);
fullBreathFeatures = getSegmentFeatures(signal(3:21), 20);

verifyEqual(testCase, shapeFeatures.inhale_smoothness, ...
    inhaleFeatures.smoothness, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.inhale_curvature, ...
    inhaleFeatures.curvature, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.inpeak_to_extrough_smoothness, ...
    transitionFeatures.smoothness, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.inpeak_to_extrough_curvature, ...
    transitionFeatures.curvature, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.inpeak_to_extrough_slope, ...
    transitionFeatures.phase2slope, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.exhale_smoothness, ...
    exhaleFeatures.smoothness, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.exhale_curvature, ...
    exhaleFeatures.curvature, 'AbsTol', 1e-12);
verifyEqual(testCase, shapeFeatures.max_flowchange, ...
    fullBreathFeatures.maxFlowchange, 'AbsTol', 1e-12);
end


function testIncompleteBreathPreservesAvailableFeatures(testCase)
signal = sin(linspace(0, 4 * pi, 50));
shapeFeatures = calculateRespiratoryShapeFeatures( ...
    signal, 20, [3, 26], [7, 30], [13, 36], [14, NaN], [21, NaN]);

verifyEqual(testCase, height(shapeFeatures), 2);
verifyFalse(testCase, isnan(shapeFeatures.inhale_smoothness(2)));
verifyFalse(testCase, ...
    isnan(shapeFeatures.inpeak_to_extrough_smoothness(2)));
verifyTrue(testCase, isnan(shapeFeatures.exhale_smoothness(2)));
verifyTrue(testCase, isnan(shapeFeatures.exhale_curvature(2)));
verifyTrue(testCase, isnan(shapeFeatures.max_flowchange(2)));
end


function testClassMethodStoresShapeTable(testCase)
signal = sin(linspace(0, 2 * pi, 40));
bmObj = breathmetrics(signal, 20, 'humanAirflow');
bmObj.baselineCorrectedRespiration = signal;
bmObj.inhaleOnsets = 2;
bmObj.inhalePeaks = 6;
bmObj.exhaleTroughs = 12;
bmObj.exhaleOnsets = 13;
bmObj.exhaleOffsets = 20;

bmObj.findShapeFeatures();

verifyClass(testCase, bmObj.shapeFeatures, 'table');
verifyEqual(testCase, height(bmObj.shapeFeatures), 1);
verifyEqual(testCase, bmObj.shapeFeatures.breath_id, int64(0));
end


function testSegmentFeatureCalculations(testCase)
signal = linspace(-1, 1, 100)'.^3 + 0.1 * sin((0:99)');
fs = 25;
pointIdx = 51;

features = getSegmentFeatures(signal, fs, pointIdx);

window = round(fs / 10);
kernel = ones(window, 1) / window;
expectedSmooth = -mean(abs(conv(signal, kernel, 'same') - signal));
verifyEqual(testCase, features.smoothness, expectedSmooth, ...
    'AbsTol', 1e-12);

halfWindow = round(0.05 * numel(signal) / 2);
pointSegment = signal( ...
    (pointIdx - halfWindow):(pointIdx + halfWindow - 1));
pointSegmentFeatures = getSegmentFeatures(pointSegment, fs);
verifyEqual(testCase, features.smoothnessAroundPoint, ...
    pointSegmentFeatures.smoothness, 'AbsTol', 1e-12);
verifyEqual(testCase, features.timeSymmetryAroundPoint, ...
    pointSegmentFeatures.timeSymmetry, 'AbsTol', 1e-12);
verifyEqual(testCase, features.maxFlowchange, max(diff(signal)), ...
    'AbsTol', 1e-12);
end
