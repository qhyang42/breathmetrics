function [resp, srate, info] = zlpSyntheticTrace(varargin)
%ZLPSYNTHETICTRACE  Deterministic synthetic respiration trace for the ZLP refinement / merge tests.
%
%   [resp, srate, info] = zlpSyntheticTrace()
%   [resp, srate, info] = zlpSyntheticTrace('srate', 500, 'repeats', 4)
%
%   A nasal-pressure-like trace (inhale positive, arbitrary units, about 100 units peak to peak per breath) made of
%   raised-cosine humps: regular breaths with an end-expiratory pause, interleaved with short blocks of breaths designed to
%   exercise the rules of refineInhaleOnsets_zlp and mergeIncompleteExhales_zlp:
%
%     regular   inhale 0.8 s, exhale 1.8 s, pause: the detection places the onset early on the flat lead-in (rule 'foot')
%     far       a small slow bump ahead of a quick inhale, 2 s apart (rule 'far')
%     level     a two-stage inhale whose first stage returns only part of the way (an onset riding on a larger inhale, rule 'level')
%     double    two inhale pulses with no exhale between them (merge step 3: joined)
%     rebound   a deep exhale, a small rebound pulse, a flat pause, then the inhale (merge keep rule: the joining inhale starts
%               the breath)
%     lead      an exhale, a small bump, a long rest before the next inhale (merge step 2, lead bump)
%     blip      a small bump in a long pause after an exhale (merge step 1, pause blip)
%
%   No random numbers are used except a fixed-seed measurement noise (the global random stream is restored).
%
%   Name-value options: 'srate' (500), 'repeats' (4, instances of every special pattern), 'regularBetween' (4, regular breaths
%   between instances), 'noise' (0.4, SD of the measurement noise in units).
%
%   info: .events - struct array (type, tStart [s]) in time order for every special instance; .duration [s]; .srate.

    p = inputParser;
    p.addParameter('srate', 500);
    p.addParameter('repeats', 4);
    p.addParameter('regularBetween', 4);
    p.addParameter('noise', 0.4);
    p.parse(varargin{:});
    srate = p.Results.srate; nRep = p.Results.repeats; nReg = p.Results.regularBetween;

    % patterns: rows [start duration amplitude] relative to the pattern start, and the time the pattern occupies
    pat.regular = {[0 0.8 1; 0.8 1.8 -0.6], 4.0};
    pat.far     = {[0 1.8 -0.6; 3.8 1.0 0.25; 5.8 0.8 1; 6.6 1.8 -0.6], 8.9};
    pat.level   = {[0 1.0 0.7; 0.65 1.5 0.9; 2.15 1.8 -0.6], 5.5};
    pat.double  = {[0 1.0 1; 0.3 2.4 0.35; 1.5 1.0 0.8; 3.0 1.8 -0.6], 6.0};
    pat.rebound = {[0 1.2 -0.9; 1.2 0.9 0.55; 3.6 1.3 1; 4.9 1.8 -0.6], 7.5};
    pat.lead    = {[0 1.8 -0.6; 2.0 0.8 0.45; 6.0 1.3 1; 7.3 1.8 -0.6], 10.0};
    pat.blip    = {[0 1.8 -0.6; 3.0 0.5 0.4; 7.0 1.3 1; 8.3 1.8 -0.6], 11.0};
    farAmp = [0.25 0.22 0.28 0.25 0.30 0.20 0.26 0.24];                       % bump heights of the far pattern, cycled
    order = {'far', 'level', 'double', 'rebound', 'lead', 'blip'};

    comps = zeros(0, 3); t = 5; events = struct('type', {}, 'tStart', {});
    for k = 1:12                                                             % leading regular breaths
        comps = [comps; pat.regular{1} + [t 0 0]]; t = t + pat.regular{2}; %#ok<AGROW>
    end
    for q = 1:numel(order)
        for rep = 1:nRep
            P = pat.(order{q});
            if strcmp(order{q}, 'far'), P{1}(2, 3) = farAmp(1 + mod(rep - 1, numel(farAmp))); end
            comps = [comps; P{1} + [t 0 0]]; %#ok<AGROW>
            events(end + 1) = struct('type', order{q}, 'tStart', t); %#ok<AGROW>
            t = t + P{2};
            for k = 1:nReg
                comps = [comps; pat.regular{1} + [t 0 0]]; t = t + pat.regular{2}; %#ok<AGROW>
            end
        end
    end
    duration = t + 5;
    n = round(duration * srate);
    tt = (0:n - 1) / srate;
    x = zeros(1, n);
    for i = 1:size(comps, 1)
        m = tt >= comps(i, 1) & tt < comps(i, 1) + comps(i, 2);
        x(m) = x(m) + comps(i, 3) * 0.5 * (1 - cos(2 * pi * (tt(m) - comps(i, 1)) / comps(i, 2)));
    end
    x = 100 * x + 2 * sin(2 * pi * 1.15 * tt) + 5 * sin(2 * pi * tt / 37);   % cardiac ripple and a slow drift
    s = rng; rng(12345, 'twister');
    x = x + p.Results.noise * randn(1, n);
    rng(s);
    resp = x;
    info = struct('events', events, 'duration', duration, 'srate', srate);
end
