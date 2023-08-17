function [lam_e, f_e, s_post, lam_m, sig_m] = saf_est_nes(lam_init, sig_uk, Sy, r_est_dm)
Ng = size(lam_init, 2); % Number of groups, or in our case number of factors
[T, N] = size(r_est_dm);
penlty = 0:0.05:10;      % Range for regularization parameter
thres = 1e-4;           % Threshold parameter for estimate change

% bic = zeros(length(penlty), 1);
t = 0.005;
rh = 0;
maxiter = 200000;

for pp = 1:length(penlty)
    niter = 1;
    err = 1;
    bet = 0.8;
    y_0 = lam_init;
    lam_0 = lam_init; % initial estimate for the factor loadings as PCA estimate
    sig_uk_0 = sig_uk; % initial estimate for the error covariance matrix

    while(err && niter < maxiter)

        % Step 2: Compute gradient for soft-thresholding update of loadings
        
%         invA_0 = inv(lam_0 * lam_0' + sig_uk_0);
%         
%         inv_sig_uk_0 = inv(sig_uk_0); %diag(1 ./ diag(sig_uk_0));
%         c2 = inv_sig_uk_0 * lam_0;
%         invA_0 = inv_sig_uk_0 - c2 * inv(eye(Ng) + lam_0' * c2) * c2';
%         
%         gL = gradientL(lam_0, invA_0, Sy);
        
        inv_sig_uk_0 = diag(1 ./ diag(sig_uk_0)); %inv(sig_uk_0);
        c2 = repmat(diag(inv_sig_uk_0), 1, Ng) .* lam_0; %inv_sig_uk_0 * lam_0;
        isig_z_lam = c2 * inv(eye(Ng) + lam_0' * c2);
        invA_0_Sy = (inv_sig_uk_0 - isig_z_lam * c2') * Sy; %invA_0 = inv(lam_0 * lam_0' + sig_uk_0);
        
        gL = gradientL(lam_0, invA_0_Sy, isig_z_lam);
        B = lam_0 - t .* gL;
        
        y_post = soft_t(B, (t .* penlty(pp)) ./ (abs(lam_init).^rh));
        lam_post = y_post + bet .* (y_post - y_0);
        
        % Step 3: Update covariance matrix of the error term according to the EM
        % algorithm
%         sig_uk_post = diag(diag(Sy - (lam_post * lam_0') * invA_0 * Sy));

        sig_uk_post = diag(diag(Sy) - sum((lam_post * lam_0') .* invA_0_Sy', 2));
        
        niter = niter + 1;
        
        % Step 4: Check if parameter change is larger than thres
        err = sum(sum(abs(lam_post - lam_0).^2))/sum(sum(abs(lam_0).^2)) > thres | ...
            sum(sum((sig_uk_post - sig_uk_0).^2))/sum(sum((sig_uk_0).^2)) > thres;
        
        % Update previous step estimates
        y_0 = y_post;
        lam_0 = lam_post;
        sig_uk_0 = sig_uk_post;
    end
    if(niter == maxiter)
        warning('Function did not converge');
    end
    lam_m(:, :, pp) = lam_post;
    sig_m(:, :, pp) = sig_uk_post;
        
    indnz = mean(abs(lam_post)) > 1e-5; % Get index for non-zero factors
    
    % Step 5: Factor estimation based on GLS
    inv_Sig = inv(sig_uk_post);
    f_e = (inv(lam_post(:, indnz)' * inv_Sig * lam_post(:, indnz)) * ...
        lam_post(:, indnz)' * inv_Sig * r_est_dm')';
%     f_e = (f_e - repmat(mean(f_e), T,1))./repmat(std(f_e), T,1);
    
    resd_0 = r_est_dm - f_e * lam_post(:, indnz)';
    sig_uk_post = thres_resd_new(resd_0, 1, N, T);
%     sig_uk_post = diag(diag(cov(resd_0)));
    
    % Active set: Number of non-zero loadings
    kap(pp) = sum(sum(abs(lam_post) > 1e-5));
    
    % Calculate information criterion
    ll = logL_F(lam_post(:, indnz), f_e, sig_uk_post, Sy);
    bic(pp) = 2*ll + kap(pp) * (1/4) * sqrt((log(N)./N) + log(N)./(N.*T));

    if(kap(pp) == 0)
        break;
    end 
end

[~, pos1] = min(real(bic));

if kap(pos1) < 2
    pos1 = find(kap >= 2, 1, 'last');
end

lam_e = lam_m(:, :, pos1);
s_post = sig_m(:, :, pos1);

indnz = mean(abs(lam_e)) > 1e-3;
lam_e = lam_e(:, indnz);

inv_Sig = inv(s_post);
f_e = (inv(lam_e' * inv_Sig * lam_e) * lam_e' * inv_Sig * r_est_dm')';
% f_e = (lam_e' * inv_Sig * r_est_dm'/N)';
end

% Function that calculates the log Likelihood
function ll = logL_F(lam_0, f_e, sig_uk, Sy)
c1 = lam_0 * cov(f_e) * lam_0' + sig_uk;
[L, U, P] = lu(c1);
du = diag(U);
c = abs(det(P));% * prod(sign(du)));
logdet = log(c) + sum(log(abs(du)));
ll = (logdet +  trace(Sy * inv(c1)));
% ll = ll / size(lam_0, 1);
end

% Function that calculates the gradient of the likelihood

% function grad_l = gradientL(lam0, invA, Sy)
% grad_l = 2 * ((invA - invA * (Sy) * invA) * lam0);
% end

function grad_l = gradientL(lam0, invA_Sy, c3)
N = size(lam0, 1);
grad_l = 2 * (eye(N) - invA_Sy) * c3;
end
% Soft-Thresholding function
function lam_k = soft_t(z, a)
t1 = sign(z);
b = abs(z) - a;
t2 = max(0, b); %b .* (b >= 0);
lam_k = t1 .* t2;
end