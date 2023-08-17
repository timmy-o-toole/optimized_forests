% Compute forecasts based on the random walk process
function results = forecast_rw(data, ind_f_vars, m, h)
T = size(data, 1);
d = length(ind_f_vars);
err_rw = nan(m, d);

for tt = 1:m
    for ii = 1:d
        
        Y_hat = data(T-m+tt-h, ind_f_vars(ii));
        
        err_rw(tt, ii) = data(T-m+tt, ind_f_vars(ii)) - Y_hat;
    end
end
mse_rw = mean(err_rw.^2);

results.err = err_rw;
results.mse = mse_rw;
