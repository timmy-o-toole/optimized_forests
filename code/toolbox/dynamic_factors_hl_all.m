% [nfact, v_nfact, cr] = numfactors(panel, q_max, nbck, stp, c_max, penalty, cf, m, h, plot_opt)
% 
% log criterion to determine the number of dynamic factors according to 
% Hallin and Liska (2007) "Determining the Number of Factors in the General 
% Dynamic Factor Model", Journal of the American Statistical Association, 
% 102, 603-617     
%
% INPUT:    panel           :   T x n data matrix 
%                               data should be covariance stationary 
%           nfs_max     :   max number of static factors which gives a range on the upper bound on the number of dynamic factors  
%           nbck, stp       :   T x n_j subpanels are used where
%                               n_j = n - nbck : stp: n 
%                               (default value: nbck = floor(n/4), stp = 1)
%           c_max           :   c = [0:cmax] (default value: 3)
%           penalty         :   p1 = ((m/T)^0.5 + m^(-2) + n^(-1))*log(min([(T/m)^0.5;  m^2; n]))  
%                               p2 = (min([(T/m)^0.5;  m^2; n])).^(-1/2)  
%                               p3 = (min([(T/m)^0.5;  m^2; n])).^(-1)*log(min([(T/m)^0.5;  m^2; n]))
%                               (default value: 'p1')
%           cf              :   1/cf is granularity of c 
%                               (default value: 1000)
%           m               :   covariogram truncation 
%                               (default value: floor(sqrt(T)))
%           h               :   number of points in which the spectral 
%                               density is computed (default value: m)
%           plot_opt        :   option to draw the plot 
%                               (yes == 1, no == 0)(default value: 1)
%
% OUTPUT:   nfact           :   number of dynamic factors as function of c
%                               computed for n_j = n
%           v_nfact         :   variance in the number of dynamic factors
%                               as function of c and computed as the 
%                               n_j varies  
%           cr              :   values of c (needed for the plot)

function [q_opt, nfact, v_nfact, cr] = dynamic_factors_hl_all(panel, nfs_max, nbck, stp, c_max, penalty, cf, m, h, plot_opt)

%% Preliminary settings
[T,n] = size(panel);

if nargin < 2 
    disp('ERROR MESSAGE: Too few input arguments'); 
    return 
end

if nargin == 2
    nbck = min(floor(n/4), 5); % floor(n/4)
    stp = 1;
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(0.75*sqrt(T)); % floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if nargin == 3
    stp = 1;
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if nargin == 4
    c_max = 3;
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if nargin == 5   
    penalty = 'p1';
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if strcmp(penalty, 'p1') == 0 && strcmp(penalty, 'p2') == 0 && strcmp(penalty, 'p3') == 0
    disp('ERROR MESSAGE : Penalty function can only take 3 values: p1, p2 and p3');
    return
end

if nargin == 6
    cf = 1000;
    m = floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if nargin == 7
    m = floor(sqrt(T));
    h = m;
    plot_opt = 0;
end

if nargin == 8
    h = m;
    plot_opt = 0;
end

if nargin == 9
    plot_opt = 0;
end

%% Mean-standardize data
m_X = mean(panel);
s_X = std(panel);
X = (panel - ones(T,1)*m_X)./(ones(T,1)*s_X);

%% Compute the number of dynamic factors
s=0;
o_log = nan(length(n-nbck:stp:n),floor(c_max*cf),nfs_max);
for N = n-nbck:stp:n
    %disp(sprintf('subsample size %d',N));
    s = s+1;
    [~, rv] = sort(rand(n,1));                                                 % select randomly N series
    subpanel = X(1:T,rv(1:N));

    m_subpanel = mean(subpanel);
    s_subpanel = std(subpanel);
    subpanel = (subpanel - ones(T,1)*m_subpanel)./(ones(T,1)*s_subpanel);   % standardize the subpanel

    [~, D_X] = spectral(subpanel, N, h, m);                      % in this case we use spectral with q = N
    E = [D_X(:,h+1)  D_X(:,h+2:2*h+1)*2]*ones(h+1,1)/(2*h+1);               % all the n dynamic eigenvalues
    IC1 = flipud(cumsum(flipud(E)));                                        % compute information criterion
    
    for nfd = 1:nfs_max
        q_max = nfd;
        IC1_s = IC1(1:q_max+1,:);

        if strcmp(penalty, 'p1') == 1
            p = ((m/T)^0.5 + m^(-2) + N^(-1))*log(min([(T/m)^0.5;  m^2; N]))*ones(q_max+1,1);
        elseif strcmp(penalty, 'p2') == 1
            p = (min([(T/m)^0.5;  m^2; N])).^(-1/2)*ones(q_max+1,1);
        elseif strcmp(penalty, 'p3') == 1
            p = (min([(T/m)^0.5;  m^2; N])).^(-1)*log(min([(T/m)^0.5;  m^2; N]))*ones(q_max+1,1);
        end

        for c = 1:floor(c_max*cf)
            cc = c/cf;
            IC_log = log(IC1_s./N) + (0:q_max)'.*p*cc;
            rr = find((IC_log == ones(q_max+1,1)*min(IC_log))==1);              % compute minimum of IC
            o_log(s,c,nfd) = rr-1;
        end
    end
end

cr = (1:floor(c_max*cf))'/cf;
q_opt = nan(nfs_max, 1);
for nfd = 1:nfs_max
    nfact = o_log(end,:,nfd);                                                       % number of factors when N = n
    v_nfact = std(o_log(:,:,nfd));
    
    try
        f_stable = find(v_nfact == 0, 1);
        f_non_stable = find(v_nfact ~= 0, 1);
        if (f_stable < f_non_stable)
            s_stable = find(v_nfact(f_non_stable:end) == 0, 1);
            sec_stability_int = f_non_stable + s_stable-1;
            q_opt(nfd,1) = nfact(sec_stability_int);
        else
            s_non_stable = find(v_nfact(f_stable:end) ~= 0, 1);
            s_stable = find(v_nfact(f_stable+s_non_stable-1:end) == 0, 1);
            sec_stability_int = f_stable + s_non_stable + s_stable-2;
            q_opt(nfd,1) = nfact(sec_stability_int);
        end
    catch
        q_opt(nfd,1) = 0;
    end

    if (q_opt(nfd,1) == 0)
        q_opt(nfd,1) = 1;
    end
    %% Plot if needed
    if plot_opt == 1
        figure
        plot(cr,5*v_nfact,'b-')
        hold all
        plot(cr,nfact,'r-')
        xlabel('c')
        axis tight
        legend('S_c','q^{*T}_{c;n}')
        title('estimated number of factors - log criterion')
    end
end

end

% function [P_chi, D_chi, Sigma_chi] = spectral(X, q, m, h)
%
% Computes spectral decomposition for the data matrix in input.
%
% INPUT:    X               :   T x n data matrix
%                               data should be covariance stationary and
%                               mean standardized
%           q               :   dimension of the common space
%                               (i.e. number of dynamic factors)
%           m               :   covariogram truncation
%                               (default value: floor(sqrt(T)))
%           h               :   number of points in which the spectral
%                               density is computed (default value: m)
%
% OUTPUT:   P_chi           :   n x q x 2*h+1 matrix of dynamic eigenvectors
%                               associated with the q largest dynamic eigenvalues
%                               for different frequency levels
%           D_chi           :   q x 2*h+1 matrix of dynamic eigenvalues
%                               for different frequency levels
%           Sigma_chi       :   (n x n x 2*h+1) spectral density matrix
%                               of common components with the 2*h+1 density
%                               matrices for different frequency levels

function [P_chi, D_chi, Sigma_chi] = spectral(X, q, m, h)

%% Preliminary settings
[T,n] = size(X);

if nargin < 2
    disp('ERROR MESSAGE: Too few input arguments');
    return
end

if nargin == 2
    m = floor(sqrt(T));
    h = m;
end

if nargin == 3
    h = m;
end

%% Compute M covariances
M = 2*m+1;
B = Bartlett_triang(M);                                                              % Triangular window (similar Bartlett)
Gamma_k = zeros(n,n,M);
for k = 1:m+1,
    Gamma_k(:,:,m+k) = B(m+k)*(X(k:T,:))'*(X(1:T+1-k,:))/(T-k);
    Gamma_k(:,:,m-k+2) = Gamma_k(:,:,m+k)';
end

%% Compute the spectral density matrix in H points
H = 2*h+1;
Factor = exp(-sqrt(-1)*(-m:m)'*(-2*pi*h/H:2*pi/H:2*pi*h/H));
Sigma_X = zeros(n,n,H);
for j = 1:n
    Sigma_X(j,:,:) = squeeze(Gamma_k(j,:,:))*Factor;
end

%% Create output elements
P_chi = zeros(n,q,H);
D_chi = zeros(q,H);
Sigma_chi = zeros(n,n,H);

%% Compute eigenvalues and eigenvectors
%% case with q < n-1 we can use eigs, fast method
if q < n-1 
    opt.disp = 0;
    [P, D] = eigs(squeeze(Sigma_X(:,:,h+1)),q,'LM',opt);                    % frequency zero
    D_chi(:,h+1) = diag(D);
    P_chi(:,:,h+1) = P;
    Sigma_chi(:,:,h+1) = P*D*P';

    for j = 1:h
        [P, D] = eigs(squeeze(Sigma_X(:,:,j)),q,'LM',opt);                  % other frequencies
        D_chi(:,j) = diag(D);
        D_chi(:,H+1-j) = diag(D);
        P_chi(:,:,j) = P;
        P_chi(:,:,H+1-j) = conj(P);

        Sigma_chi(:,:,j) = P*D*P';
        Sigma_chi(:,:,H+1-j) = conj(P*D*P');
    end
end

%% case with q >= n-1, we must use eig, slow method
if q >= n-1 
    [P, D] = eig(squeeze(Sigma_X(:,:,h+1)));                                % frequency zero
    [D,IX] = sort((diag(D)));                                               % sort eigenvalues and eigenvectors
    D = flipud(D);
    IX = flipud(IX);
    P = P(:,IX);
    D = diag(D);
    D_chi(:,h+1) = real(diag(D));
    P_chi(:,:,h+1) = P;
    Sigma_chi(:,:,h+1) = P*D*P';

    for j = 1:h
        [P, D] = eig(squeeze(Sigma_X(:,:,j)));                              % other frequencies
        [D,IX] = sort((diag(D)));                                           % sort eigenvalues and eigenvectors
        D = flipud(D);
        IX = flipud(IX);
        P = P(:,IX);
        D = diag(D);
        D_chi(:,j) = real(diag(D));
        D_chi(:,H+1-j) = real(diag(D));
        P_chi(:,:,j) = P;
        Sigma_chi(:,:,j) = P*D*P';
        P_chi(:,:,H+1-j) = conj(P);
        Sigma_chi(:,:,H+1-j) = conj(P*D*P');
    end 
end
end

% function w = Bartlett_triang(N)
% Compute a triangular window (similar Bartlett)
% 
% INPUT
% N number of points in the window
% 
% OUTPUT
% w triangular window of size N

function w = Bartlett_triang(N)

if N<=0
    disp(('ERROR MESSAGE: number of points must be positive'));
end

N_out = 0;
w = [];

if  N == floor(N),
   N_out = N;
else
   N_out = round(N);
   disp(('WARNING MESSAGE: rounding to nearest integer'));
end

if isempty(N_out) || N_out == 0,
   w = zeros(0,1); 
   return
elseif N_out == 1,
   w = 1;   
   return
end

if rem(N_out,2)
    % It's an odd length sequence
    w = 2*(1:(N_out+1)/2)/(N_out+1);
    w = [w w((N_out-1)/2:-1:1)]';
else
    % It's even
    w = (2*(1:(N_out+1)/2)-1)/N_out;
    w = [w w(N_out/2:-1:1)]';
end



    
% [EOF] triang.m
end
