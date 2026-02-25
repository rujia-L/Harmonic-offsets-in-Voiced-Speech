function run_vowel_frame_feature_pipeline
clear; clc;


framesDir      = "C:\Users\Liam\Desktop\audio\Arkiv (kopia)\frames";
outCsv         = fullfile(framesDir, 'frames_features_robust.csv');

K               = 6;                 % number of harmonics to use
harm_orders     = (1:K).';
rpeakThr        = 0.60;              % periodicity gate
energyGate_dB   = -85;               % energy gate
tightRatio      = 0.25;              % +/- shrink ratio around f0 mode
F0min_default   = 70;                % fallback f0 range
F0max_default   = 350;

% prior strengths (specified in Hz, converted to angular frequency variance)
sigmaDeltaHz_main   = 2.0;           % 1st OMT (tighter)
sigmaDeltaHz_refine = 6.0;           % 2nd OMT (looser)
sigma2_delta_main   = (2*pi*sigmaDeltaHz_main)^2;
sigma2_delta_refine = (2*pi*sigmaDeltaHz_refine)^2;

S = dir(fullfile(framesDir, '*.wav'));
if isempty(S), error('No frame wav files under: %s', framesDir); end
nFiles = numel(S);

%% ---------------- Table schema ----------------
deltaNames = arrayfun(@(k)sprintf('Delta_n%d_Hz',k), 1:K, 'uni', false);
featNames  = {'rms_dB','zcr_Hz','rpeak','hnr_dB','specCentroid','specSpread','specTilt', ...
              'hiBand14k_ratio','H1H2_dB','H2H4_dB'};
flagNames  = {'isWeak','isOctaveLow','isOctaveHigh','deltaValid','deltaNaN'};
extraNames = {'f0_mode','F0min','F0max','f0_cand','f0_wls','f0_omt1','f0_final','cost'};
varNames   = [{'wavName','speakerID','recType','timestamp','vowel','startSec','stopSec','durationSec'}, ...
              extraNames, deltaNames, featNames, flagNames];

varTypes   = [{'string','string','string','string','string','double','double','double'}, ...
              repmat({'double'},1,numel(extraNames)), ...
              repmat({'double'},1,K), ...
              repmat({'double'},1,numel(featNames)), ...
              {'logical','logical','logical','double','double'}];

T = table('Size',[nFiles numel(varNames)], ...
          'VariableTypes',varTypes, ...
          'VariableNames',varNames);

%% ---------------- Main loop ----------------
for i = 1:nFiles
    fn    = S(i).name;
    fpath = fullfile(S(i).folder, fn);

    % parse frame name (compatible with your current pattern)
    [origBase, vowel, startSec, stopSec] = parse_frame_name(fn);
    durationSec = stopSec - startSec;
    [speakerID, recType, timestamp] = parse_wav_meta(origBase);

    % read frame
    [y, fs] = audioread(fpath);
    if size(y,2)>1, y = y(:,1); end
    y = y(:);

    % basic metadata
    T.wavName(i)     = string(fn);
    T.speakerID(i)   = string(speakerID);
    T.recType(i)     = string(recType);
    T.timestamp(i)   = string(timestamp);
    T.vowel(i)       = string(vowel);
    T.startSec(i)    = startSec;
    T.stopSec(i)     = stopSec;
    T.durationSec(i) = durationSec;

    % ---- (1) time-domain gates (ZCR, HNR) ----
    [rms_dB, zcr_Hz, rpeak, hnr_dB] = time_feats_fixed(y, fs);
    T.rms_dB(i)  = rms_dB;
    T.zcr_Hz(i)  = zcr_Hz;
    T.rpeak(i)   = rpeak;
    T.hnr_dB(i)  = hnr_dB;

    isWeak = (rms_dB < energyGate_dB) || (rpeak < rpeakThr);
    T.isWeak(i) = isWeak; % keep frames, just tag

    % ---- (2) adaptive f0 range by ACF mode ----
    try
        [F0min_use, F0max_use, f0_mode] = autoF0Range_fromAC_simple(y, fs, tightRatio, [F0min_default F0max_default]);
        if isnan(f0_mode), F0min_use=F0min_default; F0max_use=F0max_default; end
    catch
        F0min_use=F0min_default; F0max_use=F0max_default; f0_mode=NaN;
    end
    T.f0_mode(i) = f0_mode;
    T.F0min(i)   = F0min_use;
    T.F0max(i)   = F0max_use;

    % ---- (3) OMT #1 (stronger prior) for stable f0 ----
    t = (0:numel(y)-1).' / fs;
    [w0_ls, ~] = estimate_omega0_grid(y, t, harm_orders, 2*pi*[F0min_use F0max_use], 200);
    f0_init    = w0_ls/(2*pi);
    omega_init = harm_orders * 2*pi*max(f0_init, F0min_use);

    try
        [omega0_1, ~, ~, cost1] = inharmonic_est_omt_prior(y, t, harm_orders, sigma2_delta_main, omega_init);
        f0_omt1 = omega0_1/(2*pi);
    catch
        f0_omt1 = f0_init; cost1 = NaN;
    end
    T.f0_omt1(i) = f0_omt1;

    % ---- (4) de-octave with harmonic-sum over a small candidate pool ----
    candPool = unique(fold_to_range_vec([f0_omt1, 0.5*f0_omt1, 2*f0_omt1, f0_mode, 0.5*f0_mode, 2*f0_mode], F0min_use, F0max_use));
    candPool = candPool(isfinite(candPool) & candPool>0);

    [f0_best, ~] = pick_f0_by_harmonic_sum(y, fs, candPool, K);
    T.f0_cand(i) = f0_best;

    % ---- (5) OMT #2 (looser prior) for refined f0 ----
    try
        omega_init2 = harm_orders * 2*pi*f0_best;
        [omega0_2, ~, ~, cost2] = inharmonic_est_omt_prior(y, t, harm_orders, sigma2_delta_refine, omega_init2);
        f0_final = fold_scalar(omega0_2/(2*pi), F0min_use, F0max_use);
        T.cost(i) = cost2;
    catch
        f0_final = f0_best; T.cost(i) = cost1;
    end
    T.f0_final(i) = f0_final;

    % octave flags (relative to f0_mode)
    T.isOctaveLow(i)  = isfinite(f0_mode) && f0_final < 0.70*f0_mode;
    T.isOctaveHigh(i) = isfinite(f0_mode) && f0_final > 1.60*f0_mode;

    % ---- (6) spectral offset (log-spectrum parabola within harmonic bandwidths) ----
    bwBase = max(12, 0.18*max(f0_final,eps));
    bw_k   = max(bwBase, 0.06*(1:K)*f0_final); bw_k = bw_k(:).';
    [delta_spec, fpeak, amp_lin, ok] = spectral_offsets_log(y, fs, f0_final, K, bw_k); %#ok<ASGLU>

    nValid=0; nNaN=0;
    for k = 1:K
        if ~ok(k) || abs(delta_spec(k)) > bw_k(k)
            T.(deltaNames{k})(i) = NaN; nNaN=nNaN+1;
        else
            T.(deltaNames{k})(i) = delta_spec(k); nValid=nValid+1;
        end
    end
    T.deltaValid(i)=nValid;  T.deltaNaN(i)=nNaN;

    % ---- (7) other spectral and harmonic amplitude features ----
    [specCentroid,specSpread,specTilt,hi14] = spectral_feats(y, fs);
    [H1H2_dB,H2H4_dB] = harmonic_ratios(y, fs, f0_final);
    T.specCentroid(i)    = specCentroid;
    T.specSpread(i)      = specSpread;
    T.specTilt(i)        = specTilt;
    T.hiBand14k_ratio(i) = hi14;
    T.H1H2_dB(i)         = H1H2_dB;
    T.H2H4_dB(i)         = H2H4_dB;

    if mod(i,100)==0 || i==nFiles
        fprintf('[%4d/%4d] %s | f0=%.1f Hz | Δ-valid=%d/%d\n', i, nFiles, fn, f0_final, nValid, K);
    end
end

writetable(T, outCsv);
fprintf('\nSaved table -> %s\n', outCsv);
end

%% ======================== Helper functions ========================

function [F0min_use, F0max_use, f0_mode] = autoF0Range_fromAC_simple(x, fs, tightRatio, fallback)
    if nargin<4, fallback=[70 350]; end
    x = x(:)-mean(x); N=numel(x);
    r = xcorr(x,'coeff'); r=r(N:end);
    lagMin=floor(fs/500); lagMax=min(floor(fs/50), numel(r));
    if lagMax<=lagMin, f0_mode=NaN; F0min_use=fallback(1); F0max_use=fallback(2); return; end
    [~,idx] = max(r(lagMin:lagMax));
    lag = lagMin+idx-1;
    f0_mode = fs/lag;
    if ~isfinite(f0_mode) || f0_mode<=0
        F0min_use=fallback(1); F0max_use=fallback(2); f0_mode=NaN; return;
    end
    F0min_use = max(50,  (1-tightRatio)*f0_mode);
    F0max_use = min(500, (1+tightRatio)*f0_mode);
end

function [w0_best, cost_best] = estimate_omega0_grid(y, t, harm_orders, omega_range, ngrid)
    % coarse grid search on [omega_min, omega_max] using LS residual
    wmin=omega_range(1); wmax=omega_range(2);
    cand = linspace(wmin,wmax,ngrid);
    cost_best=inf; w0_best=cand(1);
    for w0 = cand
        A = exp(1i*t*(harm_orders*w0).');
        a = A\y;
        res = y - A*a;
        c   = sum(abs(res).^2);
        if c<cost_best, cost_best=c; w0_best=w0; end
    end
end

function arr = fold_to_range_vec(v,Fmin,Fmax)
    arr = arrayfun(@(x) fold_scalar(x,Fmin,Fmax), v);
end

function y = fold_scalar(x, Fmin, Fmax)
    if ~isfinite(x) || x <= 0, y = NaN; return; end
    while x < Fmin, x = x * 2;   if x > 1e5, break; end, end
    while x > Fmax, x = x / 2;   if x < 1e-5, break; end, end
    y = x;
end

% ---- log-spectrum 3-point parabola within harmonic bandwidths ----
function [deltaHz, fpeak, amp_lin, ok] = spectral_offsets_log(x, fs, f0, K, bw_vec)
    if nargin<5 || isempty(bw_vec), bw_vec = max(20, 0.3*f0)*ones(1,K); end
    if isscalar(bw_vec), bw_vec = repmat(bw_vec,1,K); end
    x = x(:).*hamming(numel(x),'periodic');
    Nfft = 2^nextpow2(max(1024, numel(x)*4));
    X = fft(x,Nfft); M = abs(X(1:Nfft/2+1));
    f = (0:Nfft/2)*(fs/Nfft);
    logM = log(M+eps);

    deltaHz = nan(1,K); fpeak = nan(1,K); amp_lin = nan(1,K); ok=false(1,K);
    for k=1:K
        fk = k*f0; bw=bw_vec(k);
        band = (f >= fk-bw) & (f <= fk+bw);
        if ~any(band), continue; end
        [~, jloc] = max(M(band));
        idx = find(band); j = idx(jloc);
        if j>1 && j<length(M)
            a = logM(j-1); b = logM(j); c = logM(j+1);
            d = 0.5*(a - c) / (a - 2*b + c + eps);  % bin shift (log-spectrum)
            fhat = f(j) + d*(f(2)-f(1));
            mhat_log = b - 0.25*(a - c)*d;
            mhat = exp(mhat_log);
        else
            fhat = f(j); mhat = M(j);
        end
        fpeak(k)   = fhat;
        amp_lin(k) = mhat;
        deltaHz(k) = fhat - fk;
        ok(k)      = true;
    end
end

% ---- harmonic-sum scoring on log-spectrum (1/k decay) ----
function [f0_best, bestScore] = pick_f0_by_harmonic_sum(y, fs, candPool, K)
    x = y(:).*hamming(numel(y),'periodic');
    Nfft = 2^nextpow2(max(1024, numel(x)*4));
    X = fft(x,Nfft); M = abs(X(1:Nfft/2+1));
    f = (0:Nfft/2)*(fs/Nfft);
    logM = log(M+eps);
    df = f(2)-f(1);

    bestScore=-inf; f0_best=candPool(1);
    for fc = candPool(:).'
        bwBase = max(12, 0.18*fc);
        w = 0;
        for k=1:K
            bwk = max(bwBase, 0.06*k*fc);
            j1 = max(1, floor((k*fc - bwk)/df)+1);
            j2 = min(numel(f), ceil((k*fc + bwk)/df)+1);
            seg = logM(j1:j2);
            if ~isempty(seg) && all(isfinite(seg))
                w = w + (max(seg) / k);
            end
        end
        if w>bestScore, bestScore=w; f0_best=fc; end
    end
end

% ---- time-domain features (ZCR in Hz & HNR) ----
function [rms_dB,zcr_Hz,rpeak,hnr_dB] = time_feats_fixed(xf, fs)
    xf = xf(:);
    rms_dB = 20*log10(rms(xf)+eps);
    zc  = sum(xf(1:end-1).*xf(2:end) < 0);
    zcr_Hz = (zc / max(numel(xf)-1,1)) * fs;   % Hz
    x0 = xf - mean(xf);
    r  = xcorr(x0,'coeff'); N = numel(x0); r = r(N:end);
    lagMin = floor(fs/500); lagMax = min(floor(fs/50), numel(r));
    if lagMax>lagMin
        rpeak = max(r(lagMin:lagMax));
    else
        rpeak = 0;
    end
    rp = max(min(rpeak,0.9999),1e-6);
    hnr_dB = 10*log10(rp/(1-rp));
end

% ---- broadband spectral features ----
function [centroid,spread,tilt,hi14ratio] = spectral_feats(xf, fs)
    xf = xf(:).*hamming(numel(xf),'periodic');
    Nfft = 2^nextpow2(max(1024, numel(xf)));
    X   = fft(xf, Nfft);
    mag = abs(X(1:floor(Nfft/2)+1));
    f   = (0:floor(Nfft/2)).' * (fs/Nfft);
    w   = mag.^2 + eps;
    centroid = sum(f.*w)/sum(w);
    spread   = sqrt( sum(((f-centroid).^2).*w) / sum(w) );
    band = (f>=1000 & f<=4000);
    if nnz(band)>10
        y = log(mag(band)+eps); x=f(band);
        p = polyfit(x, y, 1);
        tilt = p(1);
    else
        tilt = NaN;
    end
    lo = (f>=1000 & f<=4000);
    hi14ratio = sum(w(lo))/sum(w);
end

function [H1H2_dB,H2H4_dB] = harmonic_ratios(xf, fs, f0)
    H1H2_dB = NaN; H2H4_dB = NaN;
    if ~isfinite(f0) || f0<=0, return; end
    xf = xf(:).*hamming(numel(xf),'periodic');
    Nfft = 2^nextpow2(max(1024, numel(xf)));
    X = fft(xf, Nfft);
    mag = abs(X(1:Nfft/2+1));
    f   = (0:Nfft/2)*(fs/Nfft);
    bw  = max(20, 0.1*f0);
    getPk = @(fc) max(mag(f>=fc-bw & f<=fc+bw), [], 'omitnan');
    A1 = getPk(1*f0); A2 = getPk(2*f0); A3 = getPk(3*f0); A4 = getPk(4*f0); %#ok<NASGU>
    if ~isempty(A1) && ~isempty(A2), H1H2_dB = 20*log10((A1+eps)/(A2+eps)); end
    if ~isempty(A2) && ~isempty(A4), H2H4_dB = 20*log10((A2+eps)/(A4+eps)); end
end

% ---- filename parsing compatible with 'A_..._3.240-3.280_40ms.wav' ----
function [origBase, vowel, startSec, stopSec] = parse_frame_name(fname)
    [~, base, ~] = fileparts(fname);
    parts = split(base, '_');

    origBase = base; vowel = ""; startSec = NaN; stopSec = NaN;
    if numel(parts) < 2, return; end

    if ~isempty(regexp(parts{end}, '^[0-9]+ms$', 'once'))
        parts = parts(1:end-1);
    end

    isRangeTok = @(s) ~isempty(regexp(s, '^\d+(\.\d+)?-\d+(\.\d+)?$', 'once'));
    seInd = find(cellfun(isRangeTok, parts), 1, 'last');
    if isempty(seInd), return; end
    ss = split(parts{seInd}, '-');
    startSec = str2double(ss{1});
    stopSec  = str2double(ss{2});

    vInd = [];
    for k = seInd-1:-1:1
        m = regexp(parts{k}, '^v(.+)$', 'tokens', 'once');
        if ~isempty(m)
            vowel = string(m{1});
            vInd = k;
            break;
        end
    end

    idxInd = [];
    for k = seInd-1:-1:1
        if ~isempty(regexp(parts{k}, '^\d+$', 'once'))
            idxInd = k; break;
        end
    end

    baseEnd = seInd - 1;
    if ~isempty(idxInd), baseEnd = min(baseEnd, idxInd - 1); end
    if ~isempty(vInd),   baseEnd = min(baseEnd, vInd   - 1); end

    while baseEnd >= 1 && any(strcmpi(parts{baseEnd}, {'pp','pproc','pre','proc'}))
        baseEnd = baseEnd - 1;
    end

    if baseEnd >= 1
        origBase = strjoin(parts(1:baseEnd), '_');
    end
end

function [speakerID, recType, timestamp] = parse_wav_meta(baseName)
    speakerID = "UNK"; recType = "UNK"; timestamp = "0";
    tok = regexp(baseName, '^A_([A-Za-z0-9]+)_([at])_([0-9]{8,})', 'tokens', 'once');
    if ~isempty(tok)
        speakerID = string(tok{1});
        recType   = string(tok{2});
        timestamp = string(tok{3});
    end
end

% ---- OMT with prior on inharmonicity (requires omega init) ----
function [omega0,omega_test,delta,cost_func_val] = inharmonic_est_omt_prior(y,t,harm_order_vec,sigma2_delta,omega_vec_in)
    N = length(y);
    K_vec = harm_order_vec(:);
    K = length(harm_order_vec);
    if nargin<5 || isempty(omega_vec_in)
        error('omega_vec_in must be provided (use k * 2*pi*f0 as init).');
    end
    omega_vec = omega_vec_in;

    A_func = @(omega_vec) exp(1i*t*omega_vec');
    omega0_func = @(omega_vec,amp_vec)  K_vec'*(abs(amp_vec).^2.*omega_vec)/(K_vec.^2'*abs(amp_vec).^2);
    cost_func_omt = @(omega_vec,amp_vec)  1/(2*sigma2_delta)*abs(amp_vec).^2'*(omega_vec-K_vec*omega0_func(omega_vec,amp_vec)).^2;
    cost_func_data_fit = @(omega_vec,amp_vec) N*log(sum(abs(y-A_func(omega_vec)*amp_vec).^2)+eps);
    cost_func = @(omega_vec,amp_vec) cost_func_data_fit(omega_vec,amp_vec)+cost_func_omt(omega_vec,amp_vec);
    outer_cost_func = @(param_vec) cost_func(param_vec(1:K),param_vec(K+1:2*K)+1i*param_vec(2*K+1:end));

    amp_0 = A_func(omega_vec)\y;
    param_0 = [omega_vec;real(amp_0);imag(amp_0)];

    opt_set = optimset('TolX',1e-10,'Display','off','MaxFunEvals',5000,'MaxIter',2000);
    param_test = fminsearch(outer_cost_func,param_0,opt_set);
    omega_test = param_test(1:K);
    amp_test   = param_test(K+1:2*K)+1i*param_test(2*K+1:end);

    omega0 = omega0_func(omega_test,amp_test);
    delta = omega_test-K_vec*omega0;
    cost_func_val = outer_cost_func(param_test);
end
