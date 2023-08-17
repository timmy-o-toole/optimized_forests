function results = forecast_ridge(data, ind_f_vars,m, h ,opt)
% DESCRIPTION:
% This function calculates the forecast based on ridge.
%
% -------------------------------------------------------------------------
% INPUT:
% data = dataset (one series per column)
% ind_f_vars = indices of the predictor variables
% m = number of out-of-sample periods
% h = forecast horizon
% vs_IC = variable selection information criterion
% alpha_val = alpha value for elastic net and lasso methods
% opt = options, contains various parameters used in the function
%
% OUTPUT:
% results = structure MSE, error
%
% -------------------------------------------------------------------------
% NOTES:
% The function first loads the variable selection results and corresponding information. It then selects the method name based on the current alpha value, either "EN" for elastic net or "Lasso" for lasso.
%
% The function next iterates over the target variables specified in the "vntt" field of the "opt" structure. For each target variable, the corresponding column is extracted from the input dataset to use as the response variable, while the other columns are used as predictor variables.
%
% -----------------------------------------------------------------------------

% Settings
d = length(ind_f_vars);
err_ar = nan(m, d);
T = size(data,1);

% Target Variable
for ii_var = 1:d
        
    % Out of sample Loop
    for tt = 1:m
        
        % Set corresponding index for X and Y
        idx_x = tt:T-m+tt-h;
        idx_y = tt+h:T-m+tt;
        
        %Standardize and subset
        X = data(idx_x,:);
        Y = data(idx_y, ind_f_vars(ii_var));
        
        % Standardize for lasso
        X_in = X(1:size(X,1)-1,:);
        X_mean = mean(X);
        X_std = std(X);
        X_n = (X - X_mean) ./ X_std;
        X_in_n = X_n(1:size(X,1)-1,:);
        X_out_n = X_n(end,:);
        
        Yin = Y(1:size(Y,1)-1);
        Yout = Y(size(Y,1));
        Y_in_mean = mean(Yin);
        Y_in_std = std(Yin);
        Y_in_n = (Yin - Y_in_mean) ./ Y_in_std;

        % Find the optimal Ridge Lambda using cross-validation
        alpha_ridge = 0;
        % lamda ridge min and max
        ridge_lam_range = linspace(0.001,100000,opt.LambdaCVn);

        [opt_lambda_ridge, ~] = block_cv(X_in, Yin, alpha_ridge, ridge_lam_range, opt.min_train_ratio, opt.test_size, h);


        ridge_coeffs = inv(X_in_n'*X_in_n + opt_lambda_ridge * eye(size(X_in_n,2))) * X_in_n'*Y_in_n;

        % Forecast normalized
        Y_out_hat_n = X_out_n*ridge_coeffs;
        
        % Forecast val
        Y_hat = Y_out_hat_n * Y_in_std + Y_in_mean;
        err_ar(tt, ii_var) = Yout - Y_hat;
        
    end
end

% Store results
mse_ar = mean(err_ar.^2);
results.err = err_ar;
results.mse = mse_ar;
end
