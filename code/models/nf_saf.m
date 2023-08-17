function num_saf = nf_saf(data, NF)
[T, N] = size(data);
% Sample Covariance Estimator
Sy = data' * data / T;

[evec,ev] = eig(Sy);
dg_ev = diag(ev) + 0.2;
Sy = evec * diag(dg_ev) * evec';

% Initial estimates based on PCA
[lam_0, ~, resd_0] = Static_factor_est(data, NF);
sig_e_0 = thres_resd_new(resd_0, 100, N, T);
lam_e = saf_est_nes(lam_0, sig_e_0, Sy, data);

num_saf = sum(sum(abs(lam_e) > 1e-5) > sqrt(N)) ;
if(num_saf == 0)
    num_saf = 1;
end
end