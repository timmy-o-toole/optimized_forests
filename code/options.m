% Settings file

opt.c = 1;           % include constant if set to one
opt.m = 192;         % out-of-sample periods
opt.max_AR = 6;      % maximum AR lags in AR(p) process
opt.max_F = 6;       % maximum number of lags of factors in diffusion model
opt.nf_static = 10;  % maximum possible number of static factors
opt.h = [1,3,6,12];   % forecast horizons [1, 6, 12]
opt.ic = 'bic';      % information criterion, either 'aic', 'bic', 'hq'
opt.ic_VS = "CV";   % Optimal IC for Lasso/EN Forecast ["CV", "AIC", "BIC"] and forecast_models_selected.m
opt.direct = 1;      % compute direct forecast for ar if 1, otherwise iterative.

opt.run_pretransformation = 0; % [0,1] If one, recreate the dictionary. This is done when new data is added
opt.stationarity_test = 'adf & pp'; %['adf & pp', 'pp', 'adf'] % perform stationarity test according to the most restrictive of the two tests, the pp or the adf test
opt.interpolating_method = 'spline'; % ['none', 'spline', 'factor']
opt.transformation_method = 'fred'; % ['fred', 'fred_all', 'all']
opt.preselection = 0; % [0,1] If one do subset if zeor do not subset

opt.start_date = '01.01.1975'; % Define start date
opt.end_date = '01.12.2022';   % Define end date

opt.vs_alpha = [0.5, 1]; % For Lasso and Elastic-Net
opt.min_train_ratio = 0.7; % Training sample size (Ratio from total)
opt.test_size = 5; % Test size in cross-validation
opt.LambdaCVn = 200; % Number of grids for Lambda in Lasso/EN - norm 200

opt.vn = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA"];

% opt.vn = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "DPCERA3M086SBEA", ...
%           "RETAILx", "HOUST", "M2SL", "CONSPI", "WPSFD49207", ...
%           "CMRMTSPLx", "RPI", "FEDFUNDS", "IPFUELS", "IPMANSICS", ...
%           "CLAIMSx", "CPIULFSL", "CUSR0000SAS", "PCEPI", "PPICMM"]; %"all" , "CPIAUCSL"
opt.vnt =[]; % Includes variable names transformations_name and "all"
opt.vntt = []; % Includes variable names transformations for all if "all" include




