clear
close all
clc

% =========================================================================
% DESCRIPTION 
% This script loads in a FRED-MD dataset, processes the dataset, and then
% estimates factors.
%
% -------------------------------------------------------------------------
% BREAKDOWN OF THE SCRIPT
% 
% Part 1: Load and label FRED-MD data.
%
% Part 2: Process data -- transform each series to be stationary and remove
%         outliers.
%
% Part 3: Estimate factors and compute R-squared and marginal R-squared. 
%
% -------------------------------------------------------------------------
% AUXILIARY FUNCTIONS
% List of auxiliary functions to be saved in same folder as this script.
%
%   prepare_missing() - transforms series based on given transformation
%       numbers
%
%   remove_outliers() - removes outliers
%
%   factors_em() - estimates factors
%
%   mrsq() - computes R-squared and marginal R-squared from factor 
%       estimates and factor loadings
%
% -------------------------------------------------------------------------
% NOTES
% Authors: Michael W. McCracken and Serena Ng
% Date: 9/5/2017
% Version: MATLAB 2014a
% Required Toolboxes: None
%
% -------------------------------------------------------------------------
% PARAMETERS TO BE CHANGED

% File name of desired FRED-MD vintage
csv_in='current_stat.csv';

% =========================================================================
% PART 1: LOAD AND LABEL DATA

% Load data from CSV file
dum=importdata(csv_in,';');

% Variable names
series=dum.textdata(1,2:end);

% Transformation numbers
tcode=dum.data(1,:);

% Raw data
rawdata=dum.data(2:end,:);

% Month/year of final observation
final_datevec=datevec(dum.textdata(end,1));
final_month=final_datevec(3);
final_year=final_datevec(1);

% Dates (monthly) are of the form YEAR+MONTH/12
% e.g. March 1970 is represented as 1970+3/12
% Dates go from 1959:01 to final_year:final_month (see above)
dates = (1959+1/12:1/12:final_year+final_month/12)';

% T = number of months in sample
T=size(dates,1);
rawdata=rawdata(1:T,:);

% =========================================================================
% PART 2: PROCESS DATA

% Transform raw data to be stationary using auxiliary function
% prepare_missing()
yt=prepare_missing(rawdata,tcode);

% Reduce sample to usable dates: remove first two months because some
% series have been first differenced
yt=yt(2:T,:);
dates=dates(2:T,:);

% yt=yt(3:T,:);
% dates=dates(3:T,:);

for ii = 1:size(yt, 2)
    [~, pval(ii)] = pptest(yt(:, ii), 'model', 'ARD', 'lags', 6);
%     [~, pval(ii)] = adftest(yt(~isnan(yt(:, ii)), ii), 'model', 'ARD', 'lags', 5);
end

% Remove outliers using auxiliary function remove_outliers(); see function
% or readme.txt for definition of outliers
%   data = matrix of transformed series with outliers removed
%   n = number of outliers removed from each series
% [data,n]=remove_outliers(yt);
data = yt;

save data_fred data dates series tcode;