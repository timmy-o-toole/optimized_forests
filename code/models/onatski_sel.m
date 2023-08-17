% This function implements the consistent procedure
% for the determination of the number of factors described
% in the paper "Determining the number of factors from empirical
% distribution of eigenvalues"

% Inputs: 
% 1) an N by T matrix X, which has the factor structure X=LF'+e
% 2) rmax, which is a priori maximal number of factors

% The output is the number of factors

function out=onatski_sel(X,rmax)

N=size(X,1);
T=size(X,2);

% compute the sample covariance matrix
if N<=T
    S=X*X'/T;
    minNT=N;
else
    S=X'*X/N;
    minNT=T;
end

% check that rmax is smaller enough than the sample size
if minNT<rmax+5
    disp('decrease rmax')
    rmax = rmax - 5;
%     return
end

% Step 1: 
% compute the eigenvalues of the sample covariance matrix and set j=rmax+1
[V,D]=eig(S);
D=diag(D);
[Ds,Ns]=sort(D);
D=flipdim(Ds,1);
j=rmax+1;

% Step 2:
% compute the slope beta in the regression of D(j)...D(j+4) on the constant
% and (j-1)^(2/3)...(j+3)^(2/3). Set delta=2*abs(beta)
depen=D(j:j+4,1);
indepen=[ones(5,1) (((j-1):1:(j+3)).^(2/3))'];
beta=inv(indepen'*indepen)*indepen'*depen;
delta=2*abs(beta(2,1));

% Step 3:
% compute r=max{i<=rmax s.t. D(i)-D(i+1)>=delta} or if all the differences
% are smaller than delta, set r=0. Set j=r+1. 
dif=D(1:rmax)-D(2:rmax+1);
detec=flipdim(dif<delta,1);
[nozero,zeroloc]=min(detec);
if nozero==1
    r=0;
else
    r=rmax+1-zeroloc;
end
j=r+1;

% Iterate Steps 2 and 3 several times (here 4)
for iter=1:4
    depen=D(j:j+4,1);
    indepen=[ones(5,1) (((j-1):1:(j+3)).^(2/3))'];
    beta=inv(indepen'*indepen)*indepen'*depen;
    delta=2*abs(beta(2,1));
    detec=flipdim(dif<delta,1);
    [nozero,zeroloc]=min(detec);
    if nozero==1
        r=0;
    else
        r=rmax+1-zeroloc;
    end
    j=r+1;
end
out=r;