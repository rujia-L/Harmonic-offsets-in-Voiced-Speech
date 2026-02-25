function [omega0,omega_test,delta,cost_func_val] = inharmonic_est_omt_prior(y,t,harm_order_vec,sigma2_delta,omega_vec_in)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements estimator of inharmonic signal with OMT prior, i.e., estimates
% both frequencies of sinusoids and OMT pitch.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT
% y                     -       signal vector.
% t                     -       vector of sampling times.
% harm_order_vec        -       vector of harmonic orders.
% sigma2_delta          -       variance of the inharmonicity parameters.
% 
% INPUT (optional)
% omega_vec_in          -       initial guess of frequencies for the
%                               sinusoidal components. If excluded, this is
%                               replaced by peak-picking in the
%                               periodogram.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = length(y);
K_vec = harm_order_vec(:);
K = length(harm_order_vec);

%%% If no initial guess of frequencies, replace by periodogram estimates %%
if nargin<5 || isempty(omega_vec_in)
    nfft = 2^18;
    fft_grid = (0:nfft-1)'/nfft*2*pi;
    Y = abs(fft(y,nfft)).^2;
    [PKS,LOCS,W] = findpeaks(Y,'SortStr','descend','NPeaks',K);
    
    omega_vec = sort(fft_grid(LOCS),'ascend');
else
    omega_vec = omega_vec_in;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_func = @(omega_vec) exp(1i*t*omega_vec');
omega0_func = @(omega_vec,amp_vec) ...
    K_vec'*(abs(amp_vec).^2.*omega_vec)/(K_vec.^2'*abs(amp_vec).^2);


cost_func_omt = @(omega_vec,amp_vec) ...
    1/(2*sigma2_delta)*abs(amp_vec).^2'*(omega_vec-K_vec*omega0_func(omega_vec,amp_vec)).^2;

%cost_func_data_fit = @(omega_vec,amp_vec) 1/N*sum(abs(y-A_func(omega_vec)*amp_vec).^2);
cost_func_data_fit = @(omega_vec,amp_vec) N*log(sum(abs(y-A_func(omega_vec)*amp_vec).^2));

cost_func = @(omega_vec,amp_vec) ...
    cost_func_data_fit(omega_vec,amp_vec)+cost_func_omt(omega_vec,amp_vec);

outer_cost_func = @(param_vec) cost_func(param_vec(1:K),param_vec(K+1:2*K)+1i*param_vec(2*K+1:end));

amp_0 = A_func(omega_vec)\y;

param_0 = [omega_vec;real(amp_0);imag(amp_0)];

%%%%%% Minimize cost function with respect to sinusoidal frequencies %%%%%%
opt_set = optimset('TolX',1e-10,'Display','off');
%omega_test = fminsearch(cost_func,omega_vec,opt_set);
param_test = fminsearch(outer_cost_func,param_0,opt_set);
omega_test = param_test(1:K);
amp_test = param_test(K+1:2*K)+1i*param_test(2*K+1:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Estimate of fundamental frequency and inharmonicity parameters %%%%%
omega0 = omega0_func(omega_test,amp_test);
delta = omega_test-K_vec*omega0;
cost_func_val = outer_cost_func(param_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
