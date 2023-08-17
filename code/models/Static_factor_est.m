function [L, F, resd] = Static_factor_est(data, NF)

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
% F = sqrt(T) * dXV(:, 1:NF);

% Factorloadings
% L = X' * F/T;

% F'F normalization
[XVec, XVal] = eig(X'*X);
[~, pos] = sort(diag(XVal), 'descend');

%flipping eigenvector matrix in order to have descending eigenvalues
dXV = XVec(:, pos);
% dXV = sparsePCA(X, 60, j,0,0);
% Determining Factors
L = sqrt(N) * dXV(:, 1:NF);

% Factorloadings
F = X * L/N;

resd = X - F*L';
end
