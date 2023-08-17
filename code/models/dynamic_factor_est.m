function [F_KF, F_KS, F_EM_KF, F_EM_KS, F, num_iter] = dynamic_factor_est(dat, Lam, F, fm_resd, q, r, p, method)
% Function that estimates a dynamic factor model, using three specifications

%       QML:      Max Likelihood estimates using the Expectation Maximization (EM) algorithm 
%                    (Doz, Giannone and Reichlin, 2012): Currently not used
%                         
%       TWO STEP: Principal components + Kalman filtering 
%                   Doz, Catherine & Giannone, Domenico & Reichlin, Lucrezia, 2011.
%                   "A two-step estimator for large approximate dynamic factor models based on Kalman filtering," 
%                   Journal of Econometrics, Elsevier, vol. 164(1), pages 188-205, September.
%                         
% INPUTS
% dat - observed data
% q - number of dynamic factors
% r - number of static factors
% p - lag length of factor VAR
% max_iter - maximum number of iterations in estimation
%
% OUTPUTS
% F_em -   factors from QML EM algorithm
% F_pc  -   factors using principal components
% F_kal -   factors based on two step procedure

[T,N] = size(dat);

% demean and standardize the data
x = zscore(dat);

% the number of static factors cannot be greater than the dynamic ones
if r < q
    error('q has to be less or equal to r')
end
                                                                           
Q = zeros(p*r, p*r);
Q(1:r,1:r) = eye(r);

% [Lam, F, fm_resd] = Static_factor_est(x, r);

if p > 0
    % Estimate a VAR(p) for the state equation
    var_out = varest(F', p, 0);
    H = var_out.Omega;                     % VAR covariance matrix
    if r > q    % if s is different from 0 we have q dynamic factors
        
        %extract the first q eigenvectors and eigenvalues from cov(e)
        [P, M] = eig(H);
        [M, pos] = sort(diag(M), 'descend');
        %flipping eigenvector matrix in order to have descending eigenvalues
        P = P(:, pos);
        P = P(:, 1:q);
        M = diag(M(1:q));
        
        P = P*diag(sign(P(1,:)));
        Q(1:r,1:r) = P*M*P';        % variance of the VAR shock when s>0
    else        
        Q(1:r,1:r) = H;             % variance of the VAR shock when s=0
    end
end

R = diag(diag(cov(fm_resd)));         % R diagonal

% Start values for the Kalman filter
Z = nan(T-p+1, r*p);
for kk = 0:p-1
    Z(:, r*kk+1:r*(kk+1)) = F((p-1)-kk+1:end-kk, :);
end
initF = Z(1,:)';                                                          

% VAR in companion form
A = [var_out.Phi;[kron(eye(r), eye(p-1)), zeros(r*(p-1), r)]];

% initial state covariance
initP = reshape(pinv(eye((r*p)^2)-kron(A, A))*Q(:), r*p, r*p);

Lam = [Lam, zeros(N, r*(p-1))];

if isequal(method,'2-Step')
    
    % Kalman filter
    [ft, ftm, Pt, Ptm] = K_filter(initF, initP, x, A, Lam, R, Q);
    F_KF = struct('ft', ft, 'ftm', ftm, 'Pt', Pt, 'Ptm', Ptm, 'A', A, 'Lam', Lam);
    
    % Kalman Smoother
    [f_sm, Pt_sm, Ptm_sm] = K_smoother(A, ft, ftm, Pt, Ptm, Lam, R);
    F_KS = struct('ft', f_sm, 'Pt', Pt_sm, 'Ptm', Ptm_sm, 'A', A, 'Lam', Lam);
    
elseif isequal(method,'EM')
    % EM Estimator
    
    max_iter = 1000;
    [F_EM_KF, F_EM_KS] = est_DFM_em(x, initF, initP, Lam, A, Q, R, p, q, max_iter);
    F_KF = [];
    F_KS = [];
else
        error('no method for dynamic factor estimation defined!');
end

end

% Estimate factors based on EM algorithm. Currently removed. Should be
% checked
function [F_EM_KF, F_EM_KS] = est_DFM_em(x, initF, initP, Lam, A, Q, R, p, q, max_iter)

% initialize the estimation and define ueful quantities

% ll_pre = -inf;
A_pre = -inf;
Q_pre = -inf;
R_pre = -inf;
initF_pre = -inf;
initP_pre = -inf;

num_iter = 0;

[T, N] = size(x);
r = size(A, 1);     % number of factors ( r )
y = x';
con = 0;
thres = 1e-2;

% factors estimation with the EM algorithm

    %repeat the algorithm until convergence
    while (num_iter < max_iter) && ~con
        
        %%% E step : compute the sufficient statistics 
        
        % The sufficient statistics are
        % delta = sum_t=1^T (x_t * f'_t)
        % gamma = sum_t=1^T (f_t * f'_t)   
        % gamma1 = sum_t=2^T (f_t-1 * f'_t-1)
        % gamma2 = sum_t=2^T (f_t * f'_t)
        % beta = sum_t=1^T (f_t * f'_t-1)
        % P1sum  variance of the initial state
        % x1sum  expected value of the initial state 
        
        % initialize all the sufficient statistics
        delta = zeros(N, r);      
        gamma = zeros(r, r);         
        gamma1 = zeros(r, r);     
        gamma2 = zeros(r, r);    
        beta = zeros(r, r);      
        P1sum = zeros(r, r);     
        x1sum = zeros(r, 1);      
        
        % use the function Estep to update the expected sufficient statistics
        % note that at the first iteration  we use as initial values for A, C, Q, 
        % R, initx, initV the ones computed with the principal components
        [beta_t, gamma_t, delta_t, gamma1_t, gamma2_t, x1, V1] = ...
            Estep(y, A, Lam, Q, R, initF, initP);
        
        % fix the expected sufficient statistics equal to the one computed with
        % the function Estep
        beta = beta + beta_t;
        gamma = gamma + gamma_t;
        delta = delta + delta_t;
        gamma1 = gamma1 + gamma1_t;
        gamma2 = gamma2 + gamma2_t;
        P1sum = P1sum + V1 + x1*x1';    
        x1sum = x1sum + x1;
        
        
        % update the counter for the iterations
        num_iter =  num_iter + 1;
        
        
        %%% M step 
        % compute the parameters of the model as a function of the sufficient
        % statistics (computed with the function Estep)
        
        % The formulas for the parameters derive from the maximum likelihood
        % method. In the EM algorithm we substitute in the ML estimator the 
        % sufficient statistics (computed in the E step) and then iterate the 
        % procedure till the maximization of the likelihood
        
        % C = (sum_t=1^T x_t*f'_t)* (sum_t=1^T f_t*f'_t)^-1 
        % substituting for the sufficient statistics
        
        C(:,1:r) = delta(:,1:r) * pinv(gamma(1:r,1:r));
        
        if p > 0
        
            % A = (sum_t=2^T f_t*f'_t-1)* (sum_2=1^T f_t-1*f'_t-1)^-1 
            Atemp = beta(1:r,1:r) * inv(gamma1(1:r,1:r));
            A(1:r,1:r) = Atemp;
            
            % Q = ( (sum_t=2^T f_t*f'_t) - A * (sum_2=1^T f_t-1*f'_t) )/(T-1) 
            H = (gamma2(1:r,1:r) - Atemp*beta(1:r,1:r)') / (T-1);
            if r > q
                [P, M] = eig(H);
                [M, pos] = sort(diag(M), 'descend');
                %flipping eigenvector matrix in order to have descending eigenvalues
                P = P(:, pos);
                P = P(:, 1:q);
                M = diag(M(1:q));
                P = P*diag(sign(P(1,:)));
                Q(1:r,1:r) = P*M*P';        % Q if s>0
            else 
                Q(1:r,1:r) = H;             % Q if s=0
            end
        end
                                                                                           
        
        % R = ( sum_t=1^T (x_t*x'_t) - C * f_t*x'_t) )/T 
        R = (x'*x - C*delta')/T;
        
        RR = diag(R); RR(RR<1e-7) = 1e-7; R = diag(RR); 
          
        R = diag(diag(R));                  % R diagonal

        initF = x1sum;
        initP = (P1sum - initF*initF');

        con = sum(sum((A - A_pre).^2))/sum(sum((A_pre).^2)) < thres & ...
              sum(sum((Q - Q_pre).^2))/sum(sum((Q_pre).^2)) < thres & ...
              sum(sum((R - R_pre).^2))/sum(sum((R_pre).^2)) < thres & ... 
              sum(sum((initF - initF_pre).^2))/sum(sum((initF_pre).^2)) < thres & ...
              sum(sum((initP - initP_pre).^2))/sum(sum((initP_pre).^2)) < thres;

        A_pre = A;
        Q_pre = Q;
        R_pre = R;
        initF_pre = initF;
        initP_pre = initP;
    
    end



[ft, ftm, Pt, Ptm]=K_filter(initF, initP, x, A, Lam, R, Q);

F_EM_KF = struct('ft', ft, 'ftm', ftm, 'Pt', Pt, 'Ptm', Ptm, 'A', A, 'Lam', Lam);

[f_sm, Pt_sm, Ptm_sm] = K_smoother(A, ft, ftm, Pt, Ptm, C, R);
F_EM_KS =  struct('ft', f_sm(1:r,:), 'Pt', Pt_sm, 'Ptm', Ptm_sm, 'A', A, 'Lam', Lam);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ft,ftm,Pt,Ptm] = K_filter(initF,initP,dat,A,Lam,R,Q)
% INPUTS
% dat(:,t) - the observation at time t
% A - the system matrix
% C - the observation matrix 
% Q - the system covariance 
% R - the observation covariance
% initF - the initial state (column) vector 
% initP - the initial state covariance 
% OUTPUT:
% ftm = E[F(:,t) | x(:,1:t-1)]
% Ptm = Cov[F(:,t) | x(:,1:t-1)]
% ft = E[F(:,t) | x(:,1:t)]
% Pt = Cov[F(:,t) | x(:,1:t)]
%loglik - value of the loglikelihood

[T, N]=size(dat);
r=size(A, 1);

y=dat';


ftm=[initF, zeros(r, T)];
ft=zeros(r, T);

Ptm=zeros(r, r, T);
Ptm(:,:,1)=initP;
Pt=zeros(r, r, T);

% logl = zeros(T, 1);
    for jj=1:T
        
        L = Lam * Ptm(:,:,jj) * Lam' + R;
        e = y(:,jj) - Lam * ftm(:,jj);
        ft(:,jj) = ftm(:,jj) + Ptm(:,:,jj) * Lam' * (L \ e);
        Pt(:,:,jj) = Ptm(:,:,jj) - Ptm(:,:,jj) * Lam' * (L \ Lam) * Ptm(:,:,jj); 
        ftm(:,jj+1) = A * ft(:,jj);
        Ptm(:,:,jj+1) = A * Pt(:,:,jj) * A' + Q;
        
%         S = Lam * Ptm(:,:,jj) * Lam' + R;
%         Sinv = inv(S);
%         
%         %%%%%%%%%%%%%%%%%%%%%%
%         
%         %log likelihood
%         logl(jj) = -(N * log(2*pi) + log(abs(det(S))) + trace(Sinv * (e*e')))/2;
        
    end

% loglik=sum(logl);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fT, PT, PTm] = K_smoother(A, ft, ftm, Pt, Ptm, Lam, R)
% INPUTS
% A - the system matrix
% ftm = E[F(:,t) | x(:,1:t-1)]
% Ptm = Cov[F(:,t) | x(:,1:t-1)]
% ft = E[F(:,t) | x(:,1:t)]
% Pt = Cov[F(:,t) | x(:,1:t)]
% Lam - the observation matrix 
% R - the observation covariance

% OUTPUT:
% fT = E[F(:,t) | x(:,1:T)]
% PT = Cov[F(:,t) | x(:,1:T)]
% PTm = Cov[F(:,t+1), F(:,t) | x(:,1:T)]

T = size(ft, 2);
N = size(Lam, 1);
r = size(A, 1);
Ptm = Ptm(:, :, 1:end-1);
ftm = ftm(:, 1:end-1);
J = zeros(r, r, T);

L = zeros(N, N, T);
K = zeros(r, N, T);

    for ii=1:T
        L(:,:,ii)=inv(Lam*Ptm(:,:,ii)*Lam'+R);
        K(:,:,ii)=Ptm(:,:,ii)*Lam'*L(:,:,ii);
    end


fT = [zeros(r,T-1)  ft(:,T)];
PT = zeros(r,r,T);
PTm = zeros(r,r,T);
PT(:,:,T) = Pt(:,:,T);
PTm(:,:,T) = (eye(r) - K(:,:,T) * Lam) * A * Pt(:,:,T-1);

    for jj =1:T-1
        J(:,:,jj) = Pt(:,:,jj) * A' / (Ptm(:,:,jj+1));
        fT(:,T-jj)= ft(:,T-jj) + J(:,:,T-jj) * (fT(:,T+1-jj) - ftm(:,T+1-jj));
        PT(:,:,T-jj) = Pt(:,:,T-jj) + J(:,:,T-jj) * (PT(:,:,T+1-jj) - ...
            Ptm(:,:,T+1-jj)) * J(:,:,T-jj)'; 
    end


    for jj =1:T-2
        PTm(:,:,T-jj) = Pt(:,:,T-jj) * J(:,:,T-jj-1)' + J(:,:,T-jj) * ...
            (PTm(:,:,T-jj+1) - A * Pt(:,:,T-jj)) * J(:,:,T-jj-1)';
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [beta, gamma, delta, gamma1, gamma2, x1, V1, xsmooth] = ...
    Estep(x, A, Lam, Q, R, initF, initP)

% This function computes the (expected) sufficient statistics for 
% a single Kalman filter sequence.
%
% y is the observable and x the hidden state

% INPUTS
% y(:,t) - the observation at time t
% A - the system matrix
% C - the observation matrix 
% Q - the system covariance 
% R - the observation covariance
% initx - the initial state (column) vector 
% initV - the initial state covariance 

% OUTPUTS: the expected sufficient statistics, i.e.
% beta = sum_t=1^T (x_t * x'_t-1)
% gamma = sum_t=1^T (x_t * x'_t)  
% delta = sum_t=1^T (y_t * x'_t)
% gamma1 = sum_t=2^T (x_t-1 * x'_t-1)
% gamma2 = sum_t=2^T (x_t * x'_t)
% x1  expected value of the initial state
% V1  variance of the initial state
% loglik value of the loglikelihood
% xsmooth expected value of the state

[N, T] = size(x);
r = length(A);

% use the Kalman smoother to compute 
% xsmooth = E[X(:,t) | y(:,1:T)]
% Vsmooth = Cov[X(:,t) | y(:,1:T)]
% VVsmooth = Cov[X(:,t), X(:,t-1) | y(:,1:T)] t >= 2
% loglik = sum{t=1}^T log P(y(:,t))
% [xitt,xittm,Ptt,Pttm,loglik_t]=K_filter(initF,initP,x',A,Lam,R,Q);
[xitt,xittm,Ptt,Pttm]=K_filter(initF,initP,x',A,Lam,R,Q);
[xsmooth, Vsmooth, VVsmooth]=K_smoother(A,xitt,xittm,Ptt,Pttm,Lam,R);

% compute the expected sufficient statistics
delta = zeros(N, r);
gamma = zeros(r, r);
beta = zeros(r, r);
    for t=1:T
        delta = delta + x(:,t)*xsmooth(:,t)';
        gamma = gamma + xsmooth(:,t)*xsmooth(:,t)' + Vsmooth(:,:,t);
        if t>1 
            beta = beta + xsmooth(:,t)*xsmooth(:,t-1)' + VVsmooth(:,:,t); 
        end
    end
gamma1 = gamma - xsmooth(:,T)*xsmooth(:,T)' - Vsmooth(:,:,T);
gamma2 = gamma - xsmooth(:,1)*xsmooth(:,1)' - Vsmooth(:,:,1);

x1 = xsmooth(:,1);
V1 = Vsmooth(:,:,1);

end
