function tests = testSecondaryFeatureValidity
tests = functiontests(localfunctions);
end


function setupOnce(~)
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'breathmetrics_functions'));
end


function testRejectedBreathIsExcludedFromAllSummaryFeatures(testCase)
bmObj = breathmetrics(sin(linspace(0, 8 * pi, 400)), 100, ...
    'humanAirflow');
bmObj.inhaleOnsets = [1, 101, 201, 301];
bmObj.exhaleOnsets = [51, 151, 251, 351];
bmObj.peakInspiratoryFlows = [1, 1000, 3, 5];
bmObj.troughExpiratoryFlows = [2, 2000, 4, 6];
bmObj.inhaleVolumes = [10, 10000, 30, 50];
bmObj.exhaleVolumes = [20, 20000, 40, 60];
bmObj.inhaleDurations = [1, 100, 3, 5];
bmObj.exhaleDurations = [2, 200, 4, 6];
bmObj.inhalePauseDurations = [0.5, 50, NaN, 1.5];
bmObj.exhalePauseDurations = [NaN, 60, 1, 2];
bmObj.statuses = {'valid', 'rejected', 'valid', 'valid'};

stats = getSecondaryRespiratoryFeatures(bmObj, 0);

verifyEqual(testCase, stats('Breathing Rate'), 1, 'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Peak Inspiratory Flow'), 3, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Peak Expiratory Flow'), 4, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Inhale Volume'), 30, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Exhale Volume'), 40, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Inhale Duration'), 3, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Exhale Duration'), 4, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, stats('Percent of Breaths With Inhale Pause'), ...
    2/3, 'AbsTol', 1e-12);
verifyEqual(testCase, stats('Percent of Breaths With Exhale Pause'), ...
    2/3, 'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Inhale Pause Duration'), ...
    2/3, 'AbsTol', 1e-12);
verifyEqual(testCase, stats('Average Exhale Pause Duration'), 1, ...
    'AbsTol', 1e-12);
end
