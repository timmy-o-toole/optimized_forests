function sig_e_hat = thres_resd_new(resd, C, N, T)

rate=1/sqrt(N)+sqrt((log(N))/T);

sig_e_samp = cov(resd,1);
thet_par = (resd.^2)'*(resd.^2)/T - sig_e_samp.^2;
lambda = rate * C * sqrt(thet_par);

sig_e_diag = diag(sig_e_samp);
sig_e_diag(sig_e_diag < 1e-7) = 1e-7;
sig_e_diag=diag(sqrt(sig_e_diag));
R = inv(sig_e_diag)* sig_e_samp * inv(sig_e_diag); 
M = soft_t(R, lambda);
M(1:N+1:numel(M)) = 1;
sig_e_hat = sig_e_diag*M*sig_e_diag;

end
% Soft-Thresholding function
function z_t = soft_t(z, a)
t1 = sign(z);
b = abs(z) - a;
t2 = max(0, b); %b .* (b >= 0);
z_t = t1 .* t2;
end