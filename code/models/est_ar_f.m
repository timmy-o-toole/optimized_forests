%% function that estimates an AR(1) process
% Input: data - dependent variable
%           F - estimated factors
%           p - lags of dependent variable
%           m - lags of factors
%           c - 1 include constant, else do not include constant

function [alpha_est, res] = est_ar_f(data, F, p, m, c)
T = size(data, 1);
maxorder = max(m, p);

if (c == 1)
    X = ones(T - maxorder, 1);
    for ii = 1:p
        X = [X, data(maxorder-ii+1:end-ii, 1)];
    end
    for jj = 1:m
        X = [X, F(maxorder-jj+1:end-jj, :)];
    end
else
    X = [];
    for ii = 1:p
        X = [X, data(maxorder-ii+1:end-ii, 1)];
    end
    for jj = 1:m
        X = [X, F(maxorder-jj+1:end-jj, :)];
    end
end

alpha_est = (X'*X)\X'*data(maxorder+1:end);
res = data(maxorder+1:end, 1) - X * alpha_est;
