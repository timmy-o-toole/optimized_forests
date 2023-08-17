function [PCs,ICs] = bai_ng(data, NF)
X = data;
N = size(data,2);
T = size(data,1);

PC1 = nan(NF,1);
PC2 = nan(NF,1);
PC3 = nan(NF,1);

IC1 = nan(NF,1);
IC2 = nan(NF,1);
IC3 = nan(NF,1);

%L'L normalization
[XV, XE] = svd(X'*X / T);

%flipping eigenvector matrix in order to have descending eigenvalues
dXV = XV;

% Determining Factorvector
L = sqrt(N) * dXV;
F = (X*L)/N;
sigm2 = mean(sum((X - F*L').^2 / T));
for j = 1:NF
   
L1 = L(:,1:j);

% Factors
F = (X*L1)/N;

V = mean(sum((X - F*L1').^2 / T));
%Information criteria

%PC1
PC1(j,1) = V + j * sigm2 * ((N+T)/(N*T) * log((N*T)/(N+T)));
%PC2
PC2(j,1) = V + j * sigm2 * ((N+T)/(N*T) * log(min(T,N)));
%PC3
PC3(j,1) = V + j * sigm2 * (log(min(T,N)) / (min(T,N)));

%IC1
IC1(j,1) = log(V) + j * ((N+T)/(N*T) * log((N*T)/(N+T)));
%IC2
IC2(j,1) = log(V) + j * ((N+T)/(N*T) * log(min(T,N)));
%IC3
IC3(j,1) = log(V) + j * (log(min(T,N)) / (min(T,N)));

end
[~, PCs(1,1)] = min(PC1);
[~, PCs(2, 1)] = min(PC2);
[~, PCs(3, 1)] = min(PC3);

[~, ICs(1,1)] = min(IC1);
[~, ICs(2, 1)] = min(IC2);
[~, ICs(3, 1)] = min(IC3);

end