clear; clc;
addpath([genpath(['data', filesep]), genpath(['models', filesep]), genpath(['toolbox', filesep])]);

tic
% Load settings
options

% Load data
data = load_data('current_2022.csv');

% Adjust Settings
[opt.vnt,opt.vntt] = replace_var(opt.vn, data, opt);

% Transform data
data = data_transform(data, opt.transformation_method);

% Remove Outliers
data = outlier_remove(data);

% Remove NAs / Interpolation
data = na_remove(data, opt);

% Perform stationarity test
data = stationarity_test(data, opt);

%if ~isfile(['selection', filesep, 'vs_',char(opt.transformation_method),'.mat'])
% Compute variable selection
%vs = variable_selection_rf(data, opt);
%end

% Forecast models
[fc_results] = forecast_models(data, opt);

toc 

