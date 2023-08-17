function [L, F] = Static_factor_est1(data, NF)

X = data;
T = size(X,1);
N = size(X,2);
% % L'L normalization
% [XVec, XVal] = eig(X*X');
% [~, pos] = sort(diag(XVal), 'descend');
% 
% %flipping eigenvector matrix in order to have descending eigenvalues
% dXV = XVec(:, pos);
% % dXV = sparsePCA(X, 60, j,0,0);
% % Determining Factors
% F = nan(T, NF, NF);
% for ii = 1:NF
%     % Factorloadings
%     F(:,1:ii,ii) = sqrt(T) * dXV(:, 1:ii);
% end
% F1 = sqrt(T) * dXV(:, 1:NF);
% 
% % Factorloadings
% L = X' * F1/T;

% F'F normalization
[XVec, XVal] = eig(X'*X);
[~, pos] = sort(diag(XVal), 'descend');

%flipping eigenvector matrix in order to have descending eigenvalues
dXV = XVec(:, pos);
% dXV = sparsePCA(X, 60, j,0,0);
F = nan(T, NF, NF);
% Determining Factors
for ii = 1:NF
    L = sqrt(N) * dXV(:, 1:ii);
    % Factorloadings
    F(:,1:ii,ii) = X * L/N;
end
end
