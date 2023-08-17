function opt_lags = order_criterion_f_ar(dat, F, lagF, lagAR, c, ic)

AIC = zeros(lagAR, lagF);
HQ = zeros(lagAR, lagF);
SC = zeros(lagAR, lagF);
maxorder =  max(lagAR, lagF);
r = size(F, 2);
T = size(dat, 1) - maxorder;
for p = 1:lagAR
    for m = 1:lagF
        
        y = dat(maxorder - max(p, m) + 1:end, 1);       
        [~, res] = est_ar_f(y, F(maxorder - max(p, m) + 1:end, :), p, m, c);
%         [~, res] = est_ar_f_d(y, F(maxorder - max(p, m) + 1:end, :), p, m, c, h);
        rss = mean(res.^2);
        
%         AIC(p, m) = log(rss) + (p + m*r + c) * 2/T;
%         HQ(p, m) = log(rss) + (p + m*r + c) * 2 * (log(log(T)))/T;
%         SC(p, m) = log(rss) + (p + m*r + c) * log(T)/T;
        
        AIC(p, m) = log(rss) + (p + m*r + c) * 2/T;
        HQ(p, m) = log(rss) + (p + m*r + c) * 2 * (log(log(T)))/T;
        SC(p, m) = log(rss) + (p + m*r + c) * log(T)/T;
        
    end
end
optAIC = zeros(2, 1);
optSC = zeros(2, 1);
optHQ = zeros(2, 1);

% if(lagF == 1)
%     [~, optAIC(2,1)] = min(AIC);
%     [~, optSC(2,1)] = min(SC);
%     [~, optHQ(2,1)] = min(HQ);
%     optAIC(1,1) = 1; optSC(1,1) = 1; optHQ(1,1) = 1;
% elseif(lagAR == 1)
%     [~, optAIC(1,1)] = min(AIC);
%     [~, optSC(1,1)] = min(SC);
%     [~, optHQ(1,1)] = min(HQ);
%     optAIC(2,1) = 1; optSC(2,1) = 1; optHQ(2,1) = 1;
% else
[~, optAIC(1,1)] = min(min(AIC));
[~, optAIC(2,1)] = min(AIC(:, optAIC(1,1)));

% optAIC(1,1) = optAIC(1,1)-1;
% optAIC(2,1) = optAIC(2,1);

% if(optAIC(1,1) == 0 && optAIC(2,1) == 0)
%     optAIC(1,1) = optAIC(1,1) + 1;
% end

[~, optSC(1,1)] = min(min(SC));
[~, optSC(2,1)] = min(SC(:, optSC(1,1)));

% optSC(1,1) = optSC(1,1)-1;
% optSC(2,1) = optSC(2,1);

% if(optSC(1,1) == 0 && optSC(2,1) == 0)
%     optSC(1,1) = optSC(1,1) + 1;
% end


[~, optHQ(1,1)] = min(min(HQ));
[~, optHQ(2,1)] = min(HQ(:, optHQ(1,1)));

% optHQ(1,1) = optHQ(1,1)-1;
% optHQ(2,1) = optHQ(2,1);

% if(optHQ(1,1) == 0 && optHQ(2,1) == 0)
%     optHQ(1,1) = optHQ(1,1) + 1;
% end


% end

switch ic
    case 'aic'
        % AIC = AIC - 1;
        opt_lags = optAIC;
    case 'bic'
        % SC = SC - 1;
        opt_lags = optSC;
    case 'hq'
        % HQ = HQ - 1;
        opt_lags = optHQ;
end