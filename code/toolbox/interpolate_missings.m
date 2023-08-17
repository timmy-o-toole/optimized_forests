%% Function that interpolates missing values using a factor model and EM
%  algorithm as in Stock and Watson (2002)
% Input:  X_init: Initial dataset with missing values
%         nf_max: Maximum number of factors
% Output: X_em: Iterpolated dataset with EM algorithm
%         diff_X: Difference between estimated and initial dataset
%         pos_nan: Positions of missings
%         nanLE: Leading and ending na rows

function [X_em, diff_X, pos_nan, nanLE] = interpolate_missings(X_init, nf_max)

nf_opt = "bn"; % Option for estimating number of factors; 
               % "bn" - Bai and Ng (2002), "on" - Onatski (2008)
niter_max = 1000; % Iteration in the EM algorithm

X_0 = X_init;
[T, N] = size(X_0);

% Get positions of missing values
pos_nan = isnan(X_0);

% Returns row sum for NaN values. Marks true for rows with more
% than 80% NaN
rem1=(sum(pos_nan,2)>N*0.8);
nanLead =(cumsum(rem1)==(1:T)');
nanEnd =(cumsum(rem1(end:-1:1))==(1:T)');
nanEnd = nanEnd(end:-1:1);  % Reverses nanEnd
nanLE = (nanLead | nanEnd);

% Subsets X for for
X_0(nanLE,:) = [];

% Get positions of missing values
pos_nan = isnan(X_0);

% Initialize missing values with unconditional mean
mx = repmat(mean(X_0, "omitnan"), T, 1);
X_0(pos_nan) = mx(pos_nan);

err = 1; % Initialize err to something large
niter = 1;

X_est = factor_iterpolation(X_0, nf_opt, nf_max);

while(err > 1e-6 && niter < niter_max)
    
    % Replace observations at nan positions with fitted values
    X_0(pos_nan) = X_est(pos_nan);

    X_1_est = factor_iterpolation(X_0, nf_opt, nf_max);

    err1 = X_1_est - X_est;
    err2 = X_est(:);
    
    % Calculate relative error
    err = (err1(:)'*err1(:))/(err2'*err2);
    
    % Replace previous step data matrix with new estimated one
    X_est = X_1_est;
    niter = niter + 1;

end
X_em = X_0;
diff_X = X_em - X_init;

if (niter == niter_max)
    warning('Maximum number of iterations reached in EM algorithm');
end

end

function X_1 = factor_iterpolation(X_0, nf_opt, nf_max)

T = size(X_0, 1);

% Demean and standardize data
mu_X = repmat(mean(X_0), T, 1);
std_X = repmat(std(X_0), T, 1);

X_0_std = (X_0 - mu_X) ./ std_X;

% Estimate number of factors
if strcmp(nf_opt, "bn")
    [~, ICs] = bai_ng(X_0_std, nf_max);
    nf_static = ICs(1,1);
elseif strcmp(nf_opt, "on")
    nf_static = onatski_sel(X_0_std, nf_max);
else
    nf_static = nf_max;
end

[L, F] = Static_factor_est(X_0_std, nf_static);

% Get fitted values and transform back
X_1 = (F*L') .* std_X + mu_X;
end
