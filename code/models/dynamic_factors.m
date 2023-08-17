%% Function that the determines the number of dynamic factors according to
% the method by Bai and Ng (2007)
function q = dynamic_factors(X, NF, p, max_q)
% Input:
%   X - Demeaned data matrix
%  NF - Number of static factors
% Output:
%   q - Number of dynamic factors
[T, N] = size(X);

[~, F] = Static_factor_est(X, NF);
% lag_order = info_crit_var(F', 3, 0);

[ VARout ] = varest( F', p, 0 );
e = VARout.e;
RE = sort(eig(e'*e),'descend');

D1 = zeros(max_q - 1, 1);
D2 = zeros(max_q - 1, 1);
for l = 1:max_q-1
    D1(l,1) = RE(l+1,1).^2 ./ sum(RE(1:max_q, 1).^2);
    D2(l,1) = sum(RE(l+1:max_q).^2) ./ sum(RE(1:max_q, 1).^2);
end
    k1 = sqrt(D1) < 1/(min(N^(0.4),T^(0.4)));
    k2 = sqrt(D2) < 1/(min(N^(0.4),T^(0.4)));
    
    if(isempty(k1) || sum(k1) == 0)
        q(1,1) = 1;
    else
        q(1,1) = find(k1, 1,'first');
    end
    if(isempty(k2) || sum(k2) == 0)
        q(2,1) = 1;
    else
        q(2,1) = find(k2, 1,'first');
    end

end