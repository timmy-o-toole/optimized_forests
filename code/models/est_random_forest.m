function [forecasts, errors, mse] = random_forest_forecast(X, y, horizon)
%RANDOM_FOREST_FORECAST Forecasts time series data using a random forest model
%   [FORECASTS, ERRORS, MSE] = RANDOM_FOREST_FORECAST(X, Y, HORIZON) 
%   forecasts time series data using a random forest model. The input data X
%   and target variable Y are used to train the model, and the model is then
%   used to make forecasts on the same data. The forecast horizon is
%   specified by the HORIZON input, which determines the number of steps 
%   ahead to forecast. The function returns the forecasts, forecast errors,
%   and mean squared error.

% Import necessary libraries
addpath('../libraries/')

% Define the random forest model
model = TreeBagger(100, X, y, 'Method', 'regression', ...
    'MinLeafSize', 10, 'NumPredictorsToSample', 'all', ...
    'OOBPrediction', 'on', 'OOBVarImp', 'on');

% Use the model to make forecasts
forecasts = predict(model, X);

% Calculate the forecast errors and mean squared error
errors = forecasts - y;
mse = mean(errors.^
