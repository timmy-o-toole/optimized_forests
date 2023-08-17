function p_opt = order_criterion_arp(dat, maxorder, c, ic)

AIC = zeros(maxorder, 1);
HQ = zeros(maxorder, 1);
SC = zeros(maxorder, 1);
T = size(dat, 1) - maxorder;
for p = 1:maxorder
    y = dat(maxorder - p + 1:end, 1);
    [~, ~, res] = est_arp(y, p, c);
    rss = mean(res.^2);
    AIC(p, 1) = log(rss) + (p + c) * 2/T;
    HQ(p, 1) = log(rss) + (p + c)* 2 * (log(log(T)))/T;
    SC(p, 1) = log(rss) + (p + c)* log(T)/T;
%     AIC(p+1, 1) = log(rss) + (p + c) * 2/T;
%     HQ(p+1, 1) = log(rss) + (p + c)* 2 * (log(log(T)))/T;
%     SC(p+1, 1) = log(rss) + (p + c)* log(T)/T;
end
[~, AIC] = min(AIC);
[~, HQ] = min(HQ);
[~, SC] = min(SC);

switch ic
    case 'aic'
        % AIC = AIC - 1;
        p_opt = AIC;
    case 'bic'
        % SC = SC - 1;
        p_opt = SC;
    case 'hq'
        % HQ = HQ - 1;
        p_opt = HQ;
end