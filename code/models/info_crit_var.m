% Function that computes information criteria for lag order selection for a
% VAR process
function [lag_order, ics] = info_crit_var(dat, p_max, c)
[K, T] = size(dat);
T = T - p_max;
fpe = nan(p_max, 1);
aic = nan(p_max, 1);
hq = nan(p_max, 1);
sc = nan(p_max, 1);

for ii = 1:p_max
    y = dat(:, p_max-ii+1:end);
    if ii == 0
        var_out = varest(y, ii, 1);
    else
        var_out = varest(y, ii, c);
    end
    sig_e = var_out.Omega;
    sig_e = sig_e*(size(y,2) - ii - K*ii - c)/T;
    fpe(ii, 1) = (((T + K*ii + c) / (T - K*ii - c))^K) * det(sig_e);
    aic(ii, 1) = log(det(sig_e)) + (2*ii*K^2)/T;
    hq(ii, 1) = log(det(sig_e)) + (2*ii*K^2)* (log(log(T)))/T;
    sc(ii, 1) = log(det(sig_e)) + (ii*K^2)* (log(T))/T;
end

[fpe_crit, fpe_order] = min(fpe);
[aic_crit, aic_order] = min(aic);
[hq_crit, hq_order] = min(hq);
[sc_crit, sc_order] = min(sc);

ics = [fpe_crit, aic_crit, hq_crit, sc_crit];
lag_order = struct('fpe', fpe_order, 'aic', aic_order, ...
    'hq', hq_order, 'sc', sc_order);

    