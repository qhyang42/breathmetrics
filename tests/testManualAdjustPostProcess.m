function tests = testManualAdjustPostProcess
tests = functiontests(localfunctions);
end


function setupOnce(~)
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(repoRoot);
addpath(fullfile(repoRoot, 'breathmetrics_functions'));
end


function testMovedEventsRecalculateDurationsAndVolumes(testCase)
srate = 100;
signal = 1:400;
bmObj = breathmetrics(signal, srate, 'humanAirflow');
bmObj.baselineCorrectedRespiration = signal;
bmObj.inhaleOnsets = [10, 110, 210];
bmObj.inhalePeaks = [30, 130, 230];
bmObj.exhaleOnsets = [50, 150, 250];
bmObj.exhaleTroughs = [70, 170, 270];
bmObj.inhalePauseOnsets = [NaN, NaN, NaN];
bmObj.exhalePauseOnsets = [90, 190, 290];
bmObj.peakInspiratoryFlows = [1, 1, 1];
bmObj.troughExpiratoryFlows = [-1, -1, -1];
bmObj.statuses = {'valid', 'edited', 'valid'};

bmObj.manualAdjustPostProcess();
oldInhaleVolume = bmObj.inhaleVolumes(2);
oldExhaleVolume = bmObj.exhaleVolumes(2);

% This is one of the event arrays written back when the GUI closes.
bmObj.exhaleOnsets(2) = 155;
bmObj.manualAdjustPostProcess();

verifyEqual(testCase, bmObj.inhaleOffsets(2), 154);
verifyEqual(testCase, bmObj.exhaleOffsets(2), 189);
verifyEqual(testCase, bmObj.inhaleDurations(2), 0.44, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, bmObj.exhaleDurations(2), 0.34, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, bmObj.inhaleVolumes(2), ...
    sum(abs(signal(110:154))) / srate * 1000, 'AbsTol', 1e-12);
verifyEqual(testCase, bmObj.exhaleVolumes(2), ...
    sum(abs(signal(155:189))) / srate * 1000, 'AbsTol', 1e-12);
verifyNotEqual(testCase, bmObj.inhaleVolumes(2), oldInhaleVolume);
verifyNotEqual(testCase, bmObj.exhaleVolumes(2), oldExhaleVolume);
end
