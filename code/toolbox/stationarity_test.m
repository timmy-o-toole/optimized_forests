function data = stationarity_test(data, opt) 
% =========================================================================
% DESCRIPTION:
% This function performs stationarity test, according to the input defined 
% in the settings in opt.
%
% -------------------------------------------------------------------------
% INPUT:
%           data   = dataset (one series per column)
%           opt    = options
% 
% OUTPUT:
%           data  = stationary data
%
% -------------------------------------------------------------------------
% NOTES:
%           1) The Augmented Dickey-Fuller test and the Phillips-Perron
%           test are implemented. Which one or both is applied is defined
%           in opt.stationarity_test in settings.
%
% =========================================================================
% FUNCTION:

% Turn of warnings from ADF test
warning('off','econ:adftest:InvalidStatistic')

% Cache variables that will be used in function
y = data.data_trans_outlier_and_na_removed;
N = size(data.data_trans_outlier_and_na_removed,2);
series = data.series;

    % Loop over Variables
    for ii = 1:N

       % Run Augmented Dickey-Fuller test
       stationarity_results.adf = adftest_function(y(:,ii));
       
       % Run Phillips-Perron test with optimal lag length
       stationarity_results.pp = pptest_function(y(:,ii));
 
    % Get index of stationarity series (1:reject unit root, 0: don't reject unit root.)
        switch(opt.stationarity_test)
          case 'adf & pp' % We select the most restrictive decision from the two methods. Hence the min is used
            stationarity_results.final_decisions(ii) = min([stationarity_results.adf.h_ADF_BIC; stationarity_results.pp.h_PP]);
          case 'adf' % We select decisison according to ADF BIC
            stationarity_results.final_decisions(ii) = stationarity_results.adf.h_ADF_BIC;
          case 'pp' % We select decisison according to PP
            stationarity_results.final_decisions(ii) = stationarity_results.pp.h_PP;
        end
    
    end
    
    % Only keep the variables which are stationary
    stationary_data = y(:,logical(stationarity_results.final_decisions));

    % Get series names of sationary and non-stationary series
    series_stationary = series(logical(stationarity_results.final_decisions));
    series_non_stationary = series(~logical(stationarity_results.final_decisions));
    
    % Save into structure
    data.data_trans_outlier_and_na_removed_stationary = stationary_data;
    data.series = series_stationary;
    data.series_non_stationary = series_non_stationary;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% SUBFUNCTION

function adf_results = adftest_function(series)
% =========================================================================
% DESCRIPTION:
% Perform ADF test with optimal lag selection and adjusting effective 
% sample size for fair comparison.
%
% -------------------------------------------------------------------------
% INPUT:
%           series   = input series
% 
% OUTPUT:
%          h_ADF_BIC     = stationary variables acc. ADF BIC criterion
%          lags_ADF_BIC  = optimal number of lags acc. ADF BIC criterion
%          h_ADF_AIC     = stationary variables acc. ADF AIC criterion
%          lags_ADF_AIC  = optimal number of lags acc. ADF AIC criterion%
%
% -------------------------------------------------------------------------
% NOTES:
%           1) The BIC criterion is used to determine the optimal number of
%           lags for the ADF test. However, the AIC criterion is calculated
%           but not used. Other measures could be used as well.
%
% =========================================================================
% FUNCTION:
    
    n_lags = 10; % Maximum number of lags considered for ADF test
    
    % Loop over different number of lags
    for lags_i = 0:n_lags
        [h_i, ~, ~, ~, reg_i] = adftest(series(1+n_lags-lags_i : end), 'model', 'ARD', 'lags', lags_i); % Function from FRED: [~, pval(ii)] = adftest(yt(~isnan(yt(:, ii)), ii), 'model', 'ARD', 'lags', 5);
        h_temp(lags_i+1) = h_i;

        % Save the AIC and BIC regression results to determine the optimal lag length later
        reg_temp(lags_i+1).AIC = reg_i.AIC;
        reg_temp(lags_i+1).BIC = reg_i.BIC;

    end

    % Determine which lag order results in the lowest information criterion.

    % Select lag order with lowest IC
    [~, index_bic] = min([reg_temp.BIC]);
    [~, index_aic] = min([reg_temp.AIC]);
        
    % Get index whether stationary or not for test with optimal lag order
    adf_results.h_ADF_BIC = h_temp(index_bic);
    adf_results.h_ADF_AIC = h_temp(index_aic);
    
    % Get optimal lag order
    adf_results.lags_ADF_BIC = index_bic-1;
    adf_results.lags_ADF_AIC = index_aic-1;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function pp_results = pptest_function(series)
% =========================================================================
% DESCRIPTION:
% Perform the Phillips-Perron test with optimal lag length
%
% -------------------------------------------------------------------------
% INPUT:
%           series   = input series
% 
% OUTPUT:
%          h_PP          = stationary variables acc. Phillips Perron criterion
%          lags_PP       = optimal number of lags acc. Phillips Perron
%
% -------------------------------------------------------------------------
% NOTES:
%           1) For the pp test, we need to specify the appropriate 
%              bandwidth/lag order for the long run variance. Done by 
%              using the following: Lt_q = [q * (T/100)^0.25], 
%              where q in {4,12} and T is the time series length.
%
% =========================================================================
% FUNCTION:

    % Define function parameters
    T = length(series);
    q = 4; % can also be set to 12
    Lt = floor(q * (T/100)^0.25); % Compute the optimal lag length for the pptest

    % Perform the Phillips-Perron test
    [h_pp_temp, ~, ~, ~, ~] = pptest(series, 'model', 'ARD', 'lags', Lt); % [~, pval(ii)] = pptest(yt(:, ii), 'model', 'ARD', 'lags', 6);

    % Get index whether stationary or not for test
    pp_results.h_PP = h_pp_temp;
    
    % Get lag order
    pp_results.lags_PP = Lt;
    
end
    