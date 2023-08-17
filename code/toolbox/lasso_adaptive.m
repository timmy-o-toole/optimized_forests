function [B_adaptive, FitInfo_adaptive, ridge_weights, opt_lambda_ridge] = lasso_adaptive(X, y, alpha_val, lam_range, min_train_ratio, opt, h)
    % The adaptive_lasso function calculates the adaptive Lasso coefficients and
    % fitting information for a given dataset.
    %
    % Input parameters:
    % - Xn: The normalized predictor matrix (subset of the original data)
    % - yn: The normalized response vector (subset of the original data)
    % - alpha_val: The Lasso mixing parameter (use 1 for Lasso)
    % - lam_range: A vector of Lambda values to be tested
    % - min_train_ratio: The minimum training data ratio for
    % cross-validation for ridge params
    % - opt: A structure containing options, such as `test_size`
    % - h: The horizon for the cross-validation
    %
    % Output parameters:
    % - B_adaptive: The adaptive Lasso coefficients
    % - FitInfo_adaptive: The fitting information for the adaptive Lasso


    % Find the optimal Ridge Lambda using cross-validation
    alpha_ridge = 0;
    % lamda ridge min and max
    ridge_lam_range = linspace(0.001,500,length(lam_range));

    [opt_lambda_ridge, ~] = block_cv(X, y, alpha_ridge, ridge_lam_range, min_train_ratio, opt.test_size, h);
    
    Xn = zscore(X);
    yn = zscore(y);

    % Compute Ridge coefficients using the optimal Lambda
    XprimeX = Xn' * Xn;
    XprimeY = Xn'*yn;
    ridge_coeffs = (XprimeX + opt_lambda_ridge * eye(size(Xn,2))) \ XprimeY;

    % Calculate the weights for the adaptive Lasso using Ridge coefficient estimates
    epsilon = 1e-6;
    ridge_weights = 1 ./ (abs(ridge_coeffs) + epsilon);

    % Create the weighted predictor matrix
    Xn_weighted = Xn ./ ridge_weights';

    % Compute the adaptive Lasso coefficients using the optimal Lambda
    % Do not standardize here as there would be something wrong in that
    % case
    [B_adaptive0, FitInfo_adaptive] = lasso(Xn_weighted, yn, 'Alpha', alpha_val, 'Lambda', lam_range,'Standardize',false);

    % Retransform according to Zou (2006)
    B_adaptive = B_adaptive0 ./ ridge_weights;

end
