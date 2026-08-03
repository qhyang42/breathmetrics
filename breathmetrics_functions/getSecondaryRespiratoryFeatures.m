function respirationStatistics  = getSecondaryRespiratoryFeatures( Bm, verbose )

%calculates features of respiratory data. Running this method assumes that 
% you have already derived all possible features

if nargin < 2
    verbose = 0;
end

if verbose == 1
    disp('Calculating secondary respiratory features')
end




% first find valid breaths

% edited 4/11/20
% split nBreaths into inhales and exhales to fix indexing error from 
% myPeak error in findRespiratoryExtrema
    
nInhales=length(Bm.inhaleOnsets);
nExhales=length(Bm.exhaleOnsets);
if isempty(Bm.statuses)
    validInhaleInds=1:nInhales;
    validExhaleInds=1:nExhales;
else
    
    IndexInvalid = strfind(Bm.statuses,'rejected');
    invalidBreathInds = find(not(cellfun('isempty',IndexInvalid)));
    validInhaleInds=setdiff(1:nInhales,invalidBreathInds);
    validExhaleInds=setdiff(1:nExhales,invalidBreathInds);
    if isempty(validInhaleInds)
        warndlg('No valid breaths found. If the status of a breath is set to ''rejected'', it will not be used to compute secondary features');
    end
end

nValidInhales=length(validInhaleInds);

%%% Breathing Rate %%%
% breathing rate is the sampling rate over the average number of samples 
% in between breaths.

% Only adjacent valid breaths define an uninterrupted breathing interval.
breathDiffs=[];
for i = 1:nValidInhales-1
    thisBreath=validInhaleInds(i);
    nextBreath=validInhaleInds(i+1);
    if nextBreath == thisBreath+1
        breathDiffs(end+1)=Bm.inhaleOnsets(nextBreath) - ...
            Bm.inhaleOnsets(thisBreath); %#ok<AGROW>
    end
end

meanBreathDiff = finiteMean(breathDiffs);
breathingRate = Bm.srate/meanBreathDiff;

%%% Inter-Breath Interval %%%
% inter-breath interval is the inverse of breathing rate
interBreathInterval = 1/breathingRate;

%%% Coefficient of Variation of Breathing Rate %%% 
% this describes variability in time between breaths
cvBreathingRate = finiteCV(breathDiffs, meanBreathDiff);

if strcmp(Bm.dataType,'humanAirflow') || strcmp(Bm.dataType,'rodentAirflow')
    % the following features can only be computed for airflow data
    
    %%% Peak Flow Rates %%%
    % the maximum rate of airflow at each inhale and exhale
    
    % inhales
    validInhaleFlows=excludeOutliers(Bm.peakInspiratoryFlows, validInhaleInds);
    avgMaxInhaleFlow = finiteMean(validInhaleFlows);
    
    % exhales
    validExhaleFlows=excludeOutliers(Bm.troughExpiratoryFlows, validExhaleInds);
    avgMaxExhaleFlow = finiteMean(validExhaleFlows);

    %%% Breath Volumes %%%
    % the volume of each breath is the integral of the airflow
    
    % inhales
    validInhaleVolumes=excludeOutliers(Bm.inhaleVolumes, validInhaleInds);
    avgInhaleVolume = finiteMean(validInhaleVolumes);
    
    % exhales
    validExhaleVolumes=excludeOutliers(Bm.exhaleVolumes, validExhaleInds);
    avgExhaleVolume = finiteMean(validExhaleVolumes);

    %%% Tidal volume %%%
    % tidal volume is the total air displaced by inhale and exhale
    avgTidalVolume = avgInhaleVolume + avgExhaleVolume;

    %%% Minute Ventilation %%%
    % minute ventilation is the product of respiration rate and tidal volume
    minuteVentilation = breathingRate * avgTidalVolume;

    %%% Duty Cycle %%%
    % duty cycle is the percent of each breathing cycle that was spent in
    % a phase
    
    validInhaleDurations = Bm.inhaleDurations(validInhaleInds);
    validExhaleDurations = Bm.exhaleDurations(validExhaleInds);
    validInhalePauses = Bm.inhalePauseDurations(validInhaleInds);
    validExhalePauses = Bm.exhalePauseDurations(validExhaleInds);

    avgInhaleDuration = finiteMean(validInhaleDurations);
    avgExhaleDuration = finiteMean(validExhaleDurations);
    [pctInhalePause, avgInhalePauseDuration] = ...
        pauseSummary(validInhalePauses);
    [pctExhalePause, avgExhalePauseDuration] = ...
        pauseSummary(validExhalePauses);

    inhaleDutyCycle = avgInhaleDuration / interBreathInterval;
    inhalePauseDutyCycle = avgInhalePauseDuration / interBreathInterval;
    exhaleDutyCycle = avgExhaleDuration / interBreathInterval;
    exhalePauseDutyCycle = avgExhalePauseDuration / interBreathInterval;

    CVInhaleDuration = finiteCV(validInhaleDurations, avgInhaleDuration);
    CVInhalePauseDuration = finiteCV(validInhalePauses, ...
        avgInhalePauseDuration);
    CVExhaleDuration = finiteCV(validExhaleDurations, avgExhaleDuration);
    CVExhalePauseDuration = finiteCV(validExhalePauses, ...
        avgExhalePauseDuration);

    % coefficient of variation in breath size describes variability of breath
    % sizes
    CVTidalVolume = finiteCV(validInhaleVolumes, avgInhaleVolume);
    
end


% assigning values for output

if strcmp(Bm.dataType,'humanAirflow') || strcmp(Bm.dataType,'rodentAirflow')
    keySet= {
        'Breathing Rate';
        'Average Inter-Breath Interval';
        
        'Average Peak Inspiratory Flow';
        'Average Peak Expiratory Flow';
        
        'Average Inhale Volume';
        'Average Exhale Volume';
        'Average Tidal Volume';
        'Minute Ventilation';
        
        'Duty Cycle of Inhale';
        'Duty Cycle of Inhale Pause';
        'Duty Cycle of Exhale';
        'Duty Cycle of Exhale Pause';
        
        'Coefficient of Variation of Inhale Duty Cycle';
        'Coefficient of Variation of Inhale Pause Duty Cycle';
        'Coefficient of Variation of Exhale Duty Cycle';
        'Coefficient of Variation of Exhale Pause Duty Cycle';
        
        'Average Inhale Duration';
        'Average Inhale Pause Duration';
        'Average Exhale Duration';
        'Average Exhale Pause Duration';
        
        'Percent of Breaths With Inhale Pause';
        'Percent of Breaths With Exhale Pause';
        
        'Coefficient of Variation of Breathing Rate';
        'Coefficient of Variation of Breath Volumes';
        
        };

    valueSet={
        breathingRate; 
        interBreathInterval; 
        
        avgMaxInhaleFlow; 
        avgMaxExhaleFlow; 
        
        avgInhaleVolume; 
        avgExhaleVolume; 
        avgTidalVolume; 
        minuteVentilation; 
        
        inhaleDutyCycle; 
        inhalePauseDutyCycle; 
        exhaleDutyCycle; 
        exhalePauseDutyCycle; 
        
        CVInhaleDuration;
        CVInhalePauseDuration;
        CVExhaleDuration;
        CVExhalePauseDuration;
        
        avgInhaleDuration;
        avgInhalePauseDuration;
        avgExhaleDuration;
        avgExhalePauseDuration;
        
        pctInhalePause
        pctExhalePause;
        
        cvBreathingRate; 
        CVTidalVolume;
        };
elseif strcmp(Bm.dataType,'humanBB') || strcmp(Bm.dataType,'rodentThermocouple') 
    keySet= {
        'Breathing Rate';
        'Average Inter-Breath Interval';
        'Coefficient of Variation of Breathing Rate';
        };
    valueSet={
        breathingRate; 
        interBreathInterval; 
        cvBreathingRate; 
        };
end
    
respirationStatistics = containers.Map(keySet,valueSet);

if verbose == 1
    disp('Secondary Respiratory Features')
    for this_key = 1:length(keySet)
        fprintf('%s : %0.5g', keySet{this_key},valueSet{this_key});
        fprintf('\n')
    end
end

end

function validVals=excludeOutliers(origVals,validBreathInds)
% Return finite values from valid breaths within two population SDs.
validBreathInds = validBreathInds( ...
    validBreathInds >= 1 & validBreathInds <= numel(origVals));
validVals = origVals(validBreathInds);
validVals = validVals(isfinite(validVals));
if isempty(validVals)
    return
end

valuesMean = mean(validVals);
valuesStd = std(validVals, 1);
if valuesStd == 0
    return
end

lowerBound = valuesMean - 2 * valuesStd;
upperBound = valuesMean + 2 * valuesStd;
validVals = validVals(validVals > lowerBound & validVals < upperBound);
end


function valueMean = finiteMean(values)
values = values(isfinite(values));
if isempty(values)
    valueMean = NaN;
else
    valueMean = mean(values);
end
end


function coefficient = finiteCV(values, valueMean)
values = values(isfinite(values));
if isempty(values) || ~isfinite(valueMean) || valueMean == 0
    coefficient = NaN;
else
    coefficient = std(values, 1) / valueMean;
end
end


function [fraction, average] = pauseSummary(values)
if isempty(values)
    fraction = NaN;
    average = NaN;
    return
end

presentValues = values(isfinite(values));
fraction = numel(presentValues) / numel(values);
if isempty(presentValues)
    average = 0;
else
    average = mean(presentValues) * fraction;
end
end
