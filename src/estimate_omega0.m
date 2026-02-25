function [omega0,alpha_vec] = estimate_omega0(y,t,harm_order_vec,omega_int,nbr_of_omegas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimates the pseudo-true fundamental frequency, as well as the
% pseudo-true amplitudes and phases or the harmonics of the harmonic
% approximation.
%
% NOTE: if the noisy waveform y is replaced by a noise-free waveform, the
% estimates are identical to the corresponding pseudo-true parameters.
%
% "Defining Fundamental Frequency for Almost Harmonic Signals", Elvander
% and Jakobsson, IEEE Transaction on Signal Processing vol 68, 2020.
%
% DOI: 10.1109/TSP.2020.3035466
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT
% y                     -       signal vector.
% t                     -       vector of sampling times
% harm_order_vec        -       vector of harmonic orders.
% omega_int             -       search interval for the pseudo-true 
%                               fundamental frequency.
%
% INPUT (optional) %%%%%
% nbr_of_omegas         -       initial number of grid points for the
%                               search of the pseudo-true fundamental 
%                               frequency.
%
% OUTPUT
% omega0                -       estimate of pseudo-true fundamental
%                               frequency.
% alpha_vec             -       estimated vector of complex amplitudes for
%                               the pseudo-harmonics.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<5
   nbr_of_omegas = 200; 
end

harm_order_vec = harm_order_vec(:);

A_func = @(omega) exp(1i*omega*t*harm_order_vec');

f_exact_func = @(omega) norm(y-A_func(omega)*(A_func(omega)\y))^2;

omega_high = omega_int(2);
omega_low = omega_int(1);

tol = 1e-12;
while omega_high-omega_low>tol
    omega_grid = linspace(omega_low,omega_high,nbr_of_omegas);
    val_vec = zeros(nbr_of_omegas,1);
    for k_omega = 1:nbr_of_omegas
        val_vec(k_omega) = f_exact_func(omega_grid(k_omega));
    end
    [~,min_ind] = min(val_vec);
    omega_high = omega_grid(min(nbr_of_omegas,min_ind+1));
    omega_low = omega_grid(max(1,min_ind-1));
    nbr_of_omegas = 50;
end
omega0 = 0.5*(omega_high+omega_low);
alpha_vec = A_func(omega0)\y;

end
