function [err_df_f, err_df_f_sm] = fcast_f_df(data, F, tt, m, h, ind_f_vars)

T = length(data);
Y = data(1:T-m+tt-1, 1);

F_pred = [F.ft(:,end-h+1), zeros(size(F.ft, 1),h)];

for ii = 1:h
   F_pred(:,ii+1) = F.A * F_pred(:,ii);
end


err_df_f = data(T-m+tt, 1) - F.Lam(ind_f_vars,:) * F_pred(:,end);
err_df_f_sm = data(T-m+tt, 1) - (F.Lam(ind_f_vars,:) * F_pred(:,end) * std(Y) + mean(Y));