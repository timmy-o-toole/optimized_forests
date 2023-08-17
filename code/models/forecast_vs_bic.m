function results = forecast_vs_bic(data,data_series, ind_f_vars,m, h, alpha_val ,opt)
% DESCRIPTION:
% This function calculates the forecast of the lasso function based on preoptimized lambda values. The variable_selection.m function must be run beforehand to generate all possible parameter sets containing the necessary parameters for this function.
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
vs_IC = "BIC";
d = length(ind_f_vars);
err_ar = nan(m, d);
T = size(data,1);

% Load VS for optimal Lambda
vs_sel = load(['selection', filesep, 'vs_',opt.start_date(end-1:end),'_', char(opt.transformation_method), '.mat'], 'vs');

% Name str for selection method
if alpha_val == 0.5
    valid_m_name = genvarname('EN');
elseif  alpha_val == 1
    valid_m_name = genvarname('Lasso');
else
    warning("Alpha is neither 0.5 nor 1.0")
end

% Target Variable
for ii_var = 1:d
    
    % Define target string names and org name
    y_string_trans = data_series(ind_f_vars(ii_var)); %'INDPRO_dif_log'
    y_string = char(strtok(y_string_trans, '_')); %"INDPRO"
    
    % Define data


    % Load best Lambda for target variable.
    optimal_lambda_all = vs_sel.vs.(y_string).(['h',num2str(h)]).(valid_m_name).(vs_IC).Lambda;
    optimal_lambda = optimal_lambda_all(end-m+1:end);
    
    % Out of sample Loop
    for tt = 1:m
        
        % Set corresponding index for X and Y
        idx_x = tt:T-m+tt-h;
        idx_y = tt+h:T-m+tt;
        
        %Standardize and subset
        X = data(idx_x,:);
        Y = data(idx_y, ind_f_vars(ii_var));
        
        % Standardize for lasso
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
        
        % Compute lasso function for different lambdas
        [B, FitInfo] = lasso(X_in_n,Y_in_n,'Alpha',alpha_val,'Lambda',optimal_lambda(tt));
        
        % Forecast normalized
        Y_out_hat_n = FitInfo.Intercept +  X_out_n*B;
        
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
