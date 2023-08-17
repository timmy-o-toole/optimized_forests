%% function that estimates an AR(p) process for direct forecast
% Input:        data = (Tx1) time series
%                 p = AR lag order
%                 c = if 1: include a constant
%                 h = forecast horizon
% Output: alpha_est = estimated autoregressive coefficients
%           sig_est = estimated covariance matrix
%               res = regression residuals
%             y_hat = fitted values
            

function [alpha_est, sig_est, res, y_hat] = est_arp_d(data, p, c, h)
T = size(data, 1)-p-h+1;
if (c == 1)
    X = ones(T, 1);
    for ii = 0:p-1
        X = [X, data(p-ii:end-h-ii, 1)];
    end    
else
    X = [];
    for ii = 0:p-1
        X = [X, data(p-ii:end-h-ii, 1)];
    end 
end

alpha_est = (X'*X)\X'*data(p+h:end, 1);
y_hat = X * alpha_est;
res = data(p+h:end, 1) - y_hat;
s2 = res' * res / (T - p - c);
sig_est = inv(X'*X) .* s2;
end

