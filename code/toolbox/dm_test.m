%% Function for computing Diebold-Mariano Test for equal predictive ability
% Input:  fe_model - Forecast errors of the model
%         fe_bench - Forecast errors of the benchmark model
%                h - Forecast horizon
% Output:     t_dm - Test statistic of the Diebold-Mariano Test
%          pval_dm - p-value of the test
%               dm - Mean loss differential
%            se_dm - standard errors
%          critval - critical value of test statistic

function [t_dm, pval_dm, t_dm_mod, pval_dm_mod, dm, se_dm, critval, critval_mod] = dm_test(fe_model, fe_bench, h)


% Mean-Loss differential
d_t = fe_model.^2 - fe_bench.^2;

% Regressing d_t on a constant and computing se using Newey-West estimator
m = length(d_t);
maxLag = floor(4*(m/100)^(2/9)); %(Andrews and Monohan, 1992)
X = ones(m,1);
[~, se_dm, dm] = hac(X, d_t, 'Intercept', false, 'bandwidth',...
    maxLag+1, 'display', 'off');
t_dm = dm/se_dm;
pval_dm = 2*normcdf(-abs(t_dm)); % p-value based on std. normal dist.
critval = norminv(0.975); % critical value based on std. normal dist.

t_dm_mod = sqrt((m+1-2*h+h*(h-1)/m)/m) * t_dm;
pval_dm_mod = 2*tcdf(t_dm_mod, m-1);
critval_mod = tinv(0.975, m-1);

end