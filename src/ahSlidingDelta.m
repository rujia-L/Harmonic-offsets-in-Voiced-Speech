function res = ahSlidingDelta(x,fs,frameDur,hopDur,...
                              K,harm_order_vec,omega_int,...
                              sigma2_delta,makePlots)
% ahSlidingDelta  滑窗估计 Δ_k(t) 并输出逐帧 f0
%
% INPUT
%   x               信号 (列向量)
%   fs              采样率 (Hz)
%   frameDur        帧长 (s)
%   hopDur          帧移 (s)
%   K               谐波阶数
%   harm_order_vec  谐波序号向量 (1:K)
%   omega_int       [ω_min ω_max] (rad/s) —— w0 搜索区间
%   sigma2_delta    OMT 先验方差
%   makePlots       =true 绘制 Δ 热图
%
% OUTPUT (结构体 res)
%   validFrames     有效（非 NaN）帧数
%   meanF0, stdF0   f0 均值/标准差  (Hz)
%   mu_k,  std_k    每阶 Δ_k 的均值/标准差 (Hz)
%   f0Vec           逐帧 f0 向量 (Hz; NaN 表示无效)
%   DeltaMat        K×N 帧的 Δ_k(Hz)

% ---------- 预备 ----------
frmLen  = round(frameDur*fs);
hopLen  = round(hopDur*fs);
win     = hamming(frmLen);
nFrames = floor((length(x)-frmLen)/hopLen) + 1;

DeltaMat = NaN(K,nFrames);
f0Vec    = NaN(nFrames,1);

globalRMS = rms(x);
energyThr = 20*log10(globalRMS) - 35;   % 静音阈值 (dB)

w0_prev = NaN;                          % 保存上一帧 w0  (rad/s)

%% NEW: 与 omega_int 一致的 Hz 区间（兼容男女声）
F0min = omega_int(1)/(2*pi);
F0max = omega_int(2)/(2*pi);

% ---------- 主循环 ----------
for n = 1:nFrames
    idx = (n-1)*hopLen + (1:frmLen);
    xt  = x(idx) .* win;

    % 1) 跳过能量过低帧
    if 20*log10(rms(xt)) < energyThr, continue, end

    % 2) 初值（限制八度到 [F0min,F0max]）
    t        = (0:frmLen-1).' / fs;
    [w_ls,~] = estimate_omega0(xt,t,harm_order_vec,omega_int);
    if w_ls==0, continue, end
    % 先把 w_ls 折返到范围内（段首很关键）
    f0_ls = w_ls/(2*pi);
    while f0_ls < F0min, f0_ls = f0_ls*2; end
    while f0_ls > F0max, f0_ls = f0_ls/2; end
    w_ls   = 2*pi*f0_ls;

    if ~isnan(w0_prev)
        omega_init = (1:K).' * w0_prev;   % 用上一帧
    else
        omega_init = (1:K).' * w_ls;      % 第一帧或失效 → 自相关初值（已折返）
    end

    % 3) OMT + inharmonicity 最小化
    [w0,~,delta,~] = inharmonic_est_omt_prior( ...
                       xt,t,harm_order_vec,sigma2_delta,omega_init);

    %% NEW: 八度修正（三候选最近 + 硬折返）
    f0_raw = w0/(2*pi);
    % 三候选
    cands = [f0_raw, 0.5*f0_raw, 2.0*f0_raw];
    % 选与上一帧最近
    if ~isnan(w0_prev)
        prevF0 = w0_prev/(2*pi);
        [~,ix] = min(abs(cands - prevF0));
        f0_corr = cands(ix);
    else
        f0_corr = f0_raw;
    end
    % 硬约束：折返到 [F0min,F0max]
    f0_fold = f0_corr;
    while f0_fold < F0min, f0_fold = f0_fold*2; end
    while f0_fold > F0max, f0_fold = f0_fold/2; end

    % 若修正幅度较大（>10%）或出界，则用修正后的初值重跑一次 OMT，让 delta 一致
    need_refine = (abs(f0_fold - f0_raw)/max(f0_raw,eps) > 0.10);
    if need_refine
        omega_init2 = (1:K).' * (2*pi*f0_fold);
        [w0,~,delta,~] = inharmonic_est_omt_prior( ...
                           xt,t,harm_order_vec,sigma2_delta,omega_init2);
        f0_fold = w0/(2*pi);   % 以重跑结果为准
        % 再做一次折返（极少触发）
        while f0_fold < F0min, f0_fold = f0_fold*2; end
        while f0_fold > F0max, f0_fold = f0_fold/2; end
    end

    % 最终 f0 / w0
    curF0  = f0_fold;
    w0     = 2*pi*curF0;

    % 4) 合法性检查（与 omega_int 一致）
    if curF0 < F0min || curF0 > F0max
        f0Vec(n) = NaN;
        % 不更新 w0_prev，保持上一帧
        continue
    end

    % 5) 有效帧：保存 Δ 并更新上一帧 w0
    DeltaMat(:,n) = delta / (2*pi);      % rad/s → Hz
    f0Vec(n)      = curF0;
    w0_prev       = w0;
end

% ---------- 后处理 ----------
f0Vec = medfilt1(f0Vec,3,'omitnan');      % 3 点中位数滤波

valid      = ~isnan(f0Vec);
mu_k       = mean(DeltaMat(:,valid), 2,'omitnan');
std_k      = std (DeltaMat(:,valid), 0,2,'omitnan');
meanF0_val = mean(f0Vec(valid),'omitnan');
stdF0_val  = std (f0Vec(valid),'omitnan');


% ---------- 打包输出 ----------
res.validFrames = sum(valid);
res.meanF0      = meanF0_val;
res.stdF0       = stdF0_val;
res.mu_k        = mu_k;
res.std_k       = std_k;
res.f0Vec       = f0Vec;
res.DeltaMat    = DeltaMat;
end
