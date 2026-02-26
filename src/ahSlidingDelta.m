function res = ahSlidingDelta(x,fs,frameDur,hopDur,...
                              K,harm_order_vec,omega_int,...
                              sigma2_delta,makePlots)
% ahSlidingDelta  Sliding window estimation of Δ_k(t) and outputs frame-by-frame f0
%
% INPUT
%   x               Signal (column vector)
%   fs              Sample rate (Hz)
%   frameDur        Frame duration (s)
%   hopDur          Hop duration (s)
%   K               Harmonic order
%   harm_order_vec  Harmonic order vector (1:K)
%   omega_int       [ω_min ω_max] (rad/s) —— w0 search interval
%   sigma2_delta    OMT prior variance
%   makePlots       =true to plot Δ heatmap
%
% OUTPUT (struct res)
%   validFrames     Number of valid (non-NaN) frames
%   meanF0, stdF0   f0 mean/standard deviation (Hz)
%   mu_k,  std_k    Mean/standard deviation of each order Δ_k (Hz)
%   f0Vec           Frame-by-frame f0 vector (Hz; NaN indicates invalid)
%   DeltaMat        Δ_k for K×N frames (Hz)

% ---------- Initialization ----------
frmLen  = round(frameDur*fs);
hopLen  = round(hopDur*fs);
win     = hamming(frmLen);
nFrames = floor((length(x)-frmLen)/hopLen) + 1;

DeltaMat = NaN(K,nFrames);
f0Vec    = NaN(nFrames,1);

globalRMS = rms(x);
energyThr = 20*log10(globalRMS) - 35;   % Silence threshold (dB)

w0_prev = NaN;                          % Save previous frame's w0 (rad/s)

%% NEW: Hz interval consistent with omega_int (compatible with male and female voices)
F0min = omega_int(1)/(2*pi);
F0max = omega_int(2)/(2*pi);

% ---------- Main loop ----------
for n = 1:nFrames
    idx = (n-1)*hopLen + (1:frmLen);
    xt  = x(idx) .* win;

    % 1) Skip low energy frames
    if 20*log10(rms(xt)) < energyThr, continue, end

    % 2) Initial value (limit octave to [F0min, F0max])
    t        = (0:frmLen-1).' / fs;
    [w_ls,~] = estimate_omega0(xt,t,harm_order_vec,omega_int);
    if w_ls==0, continue, end
    % First fold w_ls back into range (crucial at the beginning of the segment)
    f0_ls = w_ls/(2*pi);
    while f0_ls < F0min, f0_ls = f0_ls*2; end
    while f0_ls > F0max, f0_ls = f0_ls/2; end
    w_ls   = 2*pi*f0_ls;

    if ~isnan(w0_prev)
        omega_init = (1:K).' * w0_prev;   % Use previous frame
    else
        omega_init = (1:K).' * w_ls;      % First frame or invalid -> autocorrelation initial value (already folded)
    end

    % 3) OMT + inharmonicity minimization
    [w0,~,delta,~] = inharmonic_est_omt_prior( ...
                       xt,t,harm_order_vec,sigma2_delta,omega_init);

    %% NEW: Octave correction (closest of three candidates + hard fold)
    f0_raw = w0/(2*pi);
    % Three candidates
    cands = [f0_raw, 0.5*f0_raw, 2.0*f0_raw];
    % Choose the closest to the previous frame
    if ~isnan(w0_prev)
        prevF0 = w0_prev/(2*pi);
        [~,ix] = min(abs(cands - prevF0));
        f0_corr = cands(ix);
    else
        f0_corr = f0_raw;
    end
    % Hard constraint: fold to [F0min, F0max]
    f0_fold = f0_corr;
    while f0_fold < F0min, f0_fold = f0_fold*2; end
    while f0_fold > F0max, f0_fold = f0_fold/2; end

    % If the correction is large (>10%) or out of bounds, rerun OMT with the corrected initial value to make delta consistent
    need_refine = (abs(f0_fold - f0_raw)/max(f0_raw,eps) > 0.10);
    if need_refine
        omega_init2 = (1:K).' * (2*pi*f0_fold);
        [w0,~,delta,~] = inharmonic_est_omt_prior( ...
                           xt,t,harm_order_vec,sigma2_delta,omega_init2);
        f0_fold = w0/(2*pi);   % Based on the rerun result
        % Fold again (rarely triggered)
        while f0_fold < F0min, f0_fold = f0_fold*2; end
        while f0_fold > F0max, f0_fold = f0_fold/2; end
    end

    % Final f0 / w0
    curF0  = f0_fold;
    w0     = 2*pi*curF0;

    % 4) Validity check (consistent with omega_int)
    if curF0 < F0min || curF0 > F0max
        f0Vec(n) = NaN;
        % Do not update w0_prev, keep the previous frame
        continue
    end

    % 5) Valid frame: save Δ and update previous frame's w0
    DeltaMat(:,n) = delta / (2*pi);      % rad/s -> Hz
    f0Vec(n)      = curF0;
    w0_prev       = w0;
end

% ---------- Post-processing ----------
f0Vec = medfilt1(f0Vec,3,'omitnan');      % 3-point median filter

valid      = ~isnan(f0Vec);
mu_k       = mean(DeltaMat(:,valid), 2,'omitnan');
std_k      = std (DeltaMat(:,valid), 0,2,'omitnan');
meanF0_val = mean(f0Vec(valid),'omitnan');
stdF0_val  = std (f0Vec(valid),'omitnan');


% ---------- Pack output ----------
res.validFrames = sum(valid);
res.meanF0      = meanF0_val;
res.stdF0       = stdF0_val;
res.mu_k        = mu_k;
res.std_k       = std_k;
res.f0Vec       = f0Vec;
res.DeltaMat    = DeltaMat;
end
