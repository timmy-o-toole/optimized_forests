%The function takes in a matrix of data, X, and a specified number of lags, lag. 
% It creates new columns in the data matrix, X_lagged, by lagging the original columns by the specified number of lags. 
% The function also has an optional input, col_names, which allows for the user to provide column names for the original data matrix. 
% If column names are provided, the function also outputs new column names for the lagged matrix, new_col_names, by appending "_lag" 
% and the corresponding lag number to the original column names.

%INPUTS
% X: the original data matrix with size n_samples x n_vars.
% lag: the number of lags you want to create
% col_names: a cell array of strings representing the column names of X. It is an optional input.

%OUTPUTS
% X_lagged: the lagged data matrix. It is of size (n_samples+lag) x (n_vars*lag+n_vars)
% new_col_names: a cell array of strings representing the column names of X_lagged. It is empty if no col_names input is provided.

function [X_lagged, new_col_names] = get_lags(X, lag, col_names)

% X is the original data matrix
% lag is the number of lags you want to create
% col_names is a cell array of strings representing the column names of X

[n_samples, n_vars] = size(X);
X_lagged = X;

% Can write this code nicer by make it similar to X - for now: fuck it.
% Make col_names_lagged = col_names 
new_col_names = {};
col_names_lagged = {};

if nargin < 3
    col_names = {};
end

for ii = 1:lag
    for jj = 1:n_vars
        
        X_temp = NaN(n_samples, 1);
        
        % Add zeros for lost lag in the beginning
        X_temp(1:end,1) = vertcat(nan(ii, 1), X(ii+1:end,jj));
        
        X_lagged = [X_lagged, X_temp];
        
        % Make new colnames
        col_names_lagged = [col_names_lagged,col_names{jj}+"_lag"+num2str(ii)];
        
    end
end

new_col_names = [col_names, col_names_lagged];

end
