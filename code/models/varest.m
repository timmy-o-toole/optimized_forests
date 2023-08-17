function [ VARout ] = varest( dat, p, c )
% Function that estimates a VAR(p) process
% Inputs:
%       dat: Time series data (T+p) x K
%       p  : lag order
%       c  : 1 or 0 dummy that specifies the inclusion of a dummy
% Outputs:
%       B_hat: coefficient estimates
%       sig_e: estimate for the residual covariance matrix

T_total = size(dat, 2);
K = size(dat,1);
Y = dat(:, (p+1):end);
T = size(Y,2);
Z = [];

% get regressor matrix depending on the inclusion of an intercept
if(c == 1)
    Z = ones(1,T);
    for i = 1:p
        Z = [Z; dat(:, (p+1-i):end-i)];
    end
    secdim = K*p + 1;
else
    for i = 1:p
        Z = [Z; dat(:, (p+1-i):end-i)];
    end
    secdim = K*p;
end

% get coefficient estimates of the VAR
B_hat = Y*Z'/(Z*Z'); 

% calculate the residuals covariance matrix
sig_e = (1/(T_total - p - K*p - c)) * (Y - B_hat*Z)*(Y - B_hat*Z)'; 

% Compute unconditional variance of y
% Write VAR(p) in companion form
A = [B_hat(:, c+1:end);[kron(eye(K), eye(p-1)), zeros(K*(p-1), K)]];
sig_U = zeros(K*p, K*p);
sig_U(1:K, 1:K) = sig_e;
sig_y_hat = reshape((eye(K^2*p^2) - kron(A, A)) \ sig_U(:), K*p, K*p);
sig_y_hat = sig_y_hat(1:K, 1:K);

temp = dat(:, end:-1:end-p+1);
if(c==1)
    % Compute one-step ahead prediction
    y_hat = B_hat(:, 2:end)*temp(:) + B_hat(:, 1);
    VARout = struct('constant',B_hat(:, 1),'Phi',B_hat(:, 2:end),...
        'y_hat', y_hat, 'sig_y', sig_y_hat, 'Omega',sig_e,'p',p, ...
        'e', (Y - B_hat*Z), 'y_fit', B_hat*Z);

else
    % Compute one-step ahead prediction
    y_hat = B_hat*temp(:);
    VARout = struct('constant',[],'Phi',B_hat,'y_hat',y_hat, ...
        'Omega',sig_e, 'sig_y', sig_y_hat, 'p',p,'e', (Y - B_hat*Z), ...
        'y_fit', B_hat*Z); 
    
end

end