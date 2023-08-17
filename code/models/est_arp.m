%% function that estimates an AR(p) process
% Input:        data = (Tx1) time series
%                 p = AR lag order
%                 c = if 1: include a constant
% Output: alpha_est = estimated autoregressive coefficients
%           sig_est = estimated covariance matrix
%               res = regression residuals
%             y_hat = fitted values
            

function [alpha_est, sig_est, res, y_hat] = est_arp(data, p, c)
T = size(data, 1) - p;
if (c == 1)
    X = ones(T, 1);
    for i = 1:p
        X = [X, data(p-i+1:end-i, 1)];
    end    
else
    X = [];
    for i = 1:p
        X = [X, data(p-i+1:end-i, 1)];
    end 
end

alpha_est = (X'*X)\X'*data(p+1:end, 1);
y_hat = X * alpha_est;
res = data(p+1:end, 1) - y_hat;
s2 = res' * res / (T - p - c);
sig_est = inv(X'*X) .* s2;
end

