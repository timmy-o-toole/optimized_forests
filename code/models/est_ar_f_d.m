%% function that estimates an factor augmented AR(p) process for direct forecast
% Input: data - dependent variable
%           F - estimated factors
%           p - lags of dependent variable
%           m - lags of factors
%           c - 1 include constant, else do not include constant
%           h - forecast horizon

function [alpha_est, res] = est_ar_f_d(data, F, p, m, c, h)
T = size(data, 1);
maxorder = max(m, p);

if (c == 1)
    X = ones(T - maxorder-h+1, 1);
    for ii = 0:p-1
        X = [X, data(maxorder-ii:end-h-ii, 1)];
    end
    for jj = 0:m-1
        X = [X, F(maxorder-jj:end-h-jj, :)];
    end
else
    X = [];
    for ii = 0:p-1
        X = [X, data(maxorder-ii:end-h-ii, 1)];
    end
    for jj = 0:m-1
        X = [X, F(maxorder-jj:end-h-jj, :)];
    end
end

alpha_est = (X'*X)\X'*data(maxorder+h:end);
res = data(maxorder+h:end, 1) - X * alpha_est;