function [AIC, BIC]=get_IC(yn, Xn, Intercept, DF, Lambda, B)

% =========================================================================
% DESCRIPTION:
% This function calculates Information Criteria (AIC, BIC)
%
% -------------------------------------------------------------------------
% INPUT:
%           yn        = target variable
%           Xn   = explanatory variables
%           Intercept = regression intercept
%           DF = Number of nonzero coefficients in B for each value of Lambda
%           B =  fitted least-squares regression coefficients for linear models
%           h = forecast horizon
%
% OUTPUT:
%           AIC = Akaike Information Criterion
%           BIC = Bayesian Information Criterion
%
% =========================================================================

% h = forecast horizon
for lam = 1:size(B,2) % Loop over all possible lambdas
    error = yn - (ones(length(yn),1)*Intercept(lam) + Xn*B(:,lam)); % Calculate forecast error
    n_err = sum(~isnan(error)); % Get error over all time periods
    logmssr = log(mean(error.^2, 'omitnan')); % get error measure
    AIC.values(lam) = logmssr + DF(lam)*2/n_err;
    BIC.values(lam) = logmssr + DF(lam)*log(n_err)/n_err;
end

% Get outputs
    [AIC.min, AIC.min_idx] = min(AIC.values);
    AIC.min_lambda = Lambda(AIC.min_idx);

    [BIC.min, BIC.min_idx] = min(BIC.values);
    BIC.min_lambda = Lambda(BIC.min_idx);

end