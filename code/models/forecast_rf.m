%% function that forecast fred_md data using an extending window approach
% Input: data - Data for estimation
%  ind_f_vars - Indices of variables to forecast
%           c - 1, include intercept, else do not insert
%           m - insample window size
%           h - forecast horizon
%          ic - information criterion: either, 'aic', 'bic', 'hq'
%      direct - 1, if direct forecast, otherwise iterative forecast


function results = forecast_rf(data, ind_f_vars, m, h, col_names, num_factors, num_lags, opt)
T = size(data, 1);
d = length(ind_f_vars);
err_ar = nan(m, d); % oos periods, variables
p_opt = nan(m, d); % oos periods, variables
num_trees = 500; % Medeiros default in R package
var_imp = cell(m, d); % oos periods, variables
var_names_all = cell(m, d);

% Target Variable
for ii = 1:d
    
    % Out of Sample forecast
    for tt = 1:m
        
        % Set corresponding index for X and Y
        idx_x = tt:T-m+tt-h;
        idx_y = tt+h:T-m+tt;
        
        % Add PCA and Factors to data
        [pca_c, pca_s] = pca(data(idx_x,:));
        x = [pca_s(:,1:num_factors), data(idx_x,:);];
        
        % Adjust Colnames to PCA, lags and names
        var_names_pca = cellfun(@(x) sprintf('PCA%d', x), num2cell(1:num_factors), 'UniformOutput', false);
        var_names = [var_names_pca, col_names];
        [X, var_names] = get_lags(x, num_lags,var_names);
        var_names_all{tt, ii} = var_names;
        
        % Target Data - select before get_variable_selection()
        Y = data(idx_y,ind_f_vars(ii));
        
        %Drop Lags which are NAN due to lag
        Y = Y(num_lags+1:end);
        X = X(num_lags+1:end,:);
        
        % Get the input and Training data which just the last obersvation
        Xin = X(1:size(X,1)-1,:);
        Yin = Y(1:size(Y,1)-1);
        Xout = X(size(X,1),:);
        Yout = Y(size(Y,1));
        
        % Create the random forest model
        rf = TreeBagger(num_trees,Xin,Yin,'Method','regression', 'OOBPredictorImportance', 'on');
       
        % Make predictions using the out-of-sample test data
        Y_pred = predict(rf,Xout);
        
        % Results
        err_ar(tt, ii) = Yout - Y_pred;
        
        % Store the indices of the selected variables
        % OOBPermutedVarDeltaError (MAE) vs  OOBPermutedPredictorDeltaError
        % (MSE)- source ChatGPT
        % The choice between these properties mainly depends on the error metric you prefer for evaluating the importance of predictor variables: mean squared error (MSE) for OOBPermutedPredictorDeltaError or mean absolute error (MAE) for OOBPermutedVarDeltaError.
        var_imp{tt, ii} = rf.OOBPermutedPredictorDeltaError;


    end
end

mse_ar = mean(err_ar.^2);
results.err = err_ar;
results.mse = mse_ar;
results.var_importance = var_imp;
results.var_names = var_names_all{1};


end
