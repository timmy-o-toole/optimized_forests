function vs = variable_selection(data, opt)
% DESCRIPTION:
%   This function performs variable selection on the input dataset using the Elastic Net and Lasso methods. The structure of the data is as follows:
%       (1) The dataset is defined by the amount of transformations.
%       (2) The selection methods are Elastic Net and Lasso
%       (3) The shrinkage method is applied for each selected variable.
%
% -------------------------------------------------------------------------
% INPUT:
%       data = dataset (one series per column)
%       opt = options, contains various parameters used in the function such as vs_alpha, vntt, CVn, LambdaMax, LambdaCVn etc.
%
% OUTPUT:
%       vs = structure {data_set.selection_method.target_variable}.
%            Binary matrix. 1 = is selected.
%            Matrix of size [out_of_sample_windows, number_variables -1]
%
% -------------------------------------------------------------------------
% NOTES:
%       The function starts with defining a hyperparameter and a timer using tic. Then it enters a loop that iterates over the values in the "vs_alpha" field of the "opt" structure, which specifies the alpha values to be used in the Elastic Net and Lasso methods. Within this loop, the method name is selected based on the current alpha value, either "EN" for Elastic Net or "Lasso" for Lasso.
%
%       Then, the function enters another loop that iterates over the target variables specified in the "vntt" field of the "opt" structure. For each target variable, the function extracts the corresponding column from the input dataset, and uses it as the response variable. The rest of the columns are used as the predictor variables.
%
%       Next, the function performs variable selection using the Elastic Net or Lasso method based on the current alpha value. The selection is performed on each out-of-sample window using 5-fold cross validation and the selected variables are stored in a matrix. Finally, the function returns the shrinked data, which is the dataset with only the selected variables.
%
%       The function also uses parfor loop to parallelize the process. The function stops the timer using toc and returns the shrinked data.
%
% =========================================================================
%Function
n_alpha = length(opt.vs_alpha); % Number of alphas
% opt.vntt = string(name + "+" + transformation)
target_vars = opt.vntt;
n_targets = length(target_vars); % Number of targets
m = opt.m; % Number of out-of-sample periods
total_iter = length(opt.h)*n_alpha*n_targets*m; % Number of total iterations

% Training sample size (Ratio from total)
min_train_ratio = opt.min_train_ratio;
tic

% (start 1) Selection Method=====================
% Is defined

for idx_h = 1:length(opt.h)
    h = opt.h(idx_h);

    % (start 2) Selection Method=====================
    for ii_method_val = 1:length(n_alpha)
        alpha_val = opt.vs_alpha(ii_method_val);

        % Name str for selection method
        if alpha_val == 0.5
            valid_m_name = genvarname('EN');
            alpha_idx = 1;
        else
            valid_m_name = genvarname('Lasso');
            alpha_idx = 2;
        end


        % (start 3) Shrinkage ===========================

        for ii_var = 1:n_targets

            % Define target string names and org name
            y_string_trans = target_vars(ii_var); %'INDPRO_dif_log'
            y_string = char(strtok(y_string_trans, '_')); %"INDPRO"

            % Define data
            dat = data.data_trans_outlier_and_na_removed_stationary;
            y = dat(:,logical(strcmp(data.series, y_string_trans)));
            X = dat;
            T = size(X,1);

            % Variable Selection Matrix for CV
            selections_CV_L = zeros(m, 1);
            selections_CV_Lidx = zeros(m, 1);
            selections_CV = zeros(m, size(X,2));

            % Variable Selection Matrix for AIC
            selections_AIC_L = zeros(m, 1);
            selections_AIC_Lidx = zeros(m, 1);
            selections_AIC = zeros(m, size(X,2));

            % Variable Selection Matrix for AIC
            selections_BIC_L = zeros(m, 1);
            selections_BIC_Lidx = zeros(m, 1);
            selections_BIC = zeros(m, size(X,2));


            % Run Selection for each out-of-sample window
            % Use parfor instead of for

            parfor tt = 1:m

                % subset and standardize
                Xsub = X(tt:T-m+tt-h-1,:);
                ysub = y(tt+h:T-m+tt-1);

                Xn = zscore(Xsub);
                yn = zscore(ysub);

                % Range for the lambda parameter
                % Calculate maximum Lambda for which no variable is selected
                lam_max = computeLambdaMax(Xn,yn,[],alpha_val);
                lam_ratio = 0.0001;
                lam_min = lam_ratio * lam_max;
                lam_range = linspace(lam_min,lam_max,opt.LambdaCVn);

                % Compute optimal lambda based on cross-validation
                [lamopt, lamopt_idx] = block_cv(Xsub,ysub,alpha_val,lam_range,min_train_ratio,opt.test_size,h);

                % Compute lasso function for different lambdas
                [B, FitInfo] = lasso(Xn,yn,'Alpha',alpha_val,'Lambda',lam_range);

                % idx of selected vars
                % select optimized nonzero_idx
                selections_CV(tt,:) = B(:,lamopt_idx)~= 0;
                selections_CV_L(tt) = lamopt;
                selections_CV_Lidx(tt) = lamopt_idx;

                % Select optimal lambda chosen by Information Criteria
                [AIC, BIC] = get_IC(yn, Xn, FitInfo.Intercept, FitInfo.DF, FitInfo.Lambda, B);

                selections_AIC(tt,:) = B(:,AIC.min_idx)~= 0;
                selections_AIC_L(tt) = AIC.min_lambda;
                selections_AIC_Lidx(tt) = AIC.min_idx;

                selections_BIC(tt,:) = B(:,BIC.min_idx)~= 0;
                selections_BIC_L(tt) = BIC.min_lambda;
                selections_BIC_Lidx(tt) = BIC.min_idx;
                
                iter = (find(h==opt.h)-1)*n_alpha*n_targets*m+(ii_method_val-1)*n_targets*m+(ii_var-1)*m+tt;
                if mod(iter, m*20) == 0
                    disp(['vs: ', num2str(round(iter/total_iter*100)), '%']);
                end
            end


            % (End 3) Set Each Variable=================
            % Store the selected indices for CV
            target.CV.S = selections_CV; % sparsify this // could use cell aswell with selections
            target.CV.Lambda = selections_CV_L; % Lasso/EN lamdas per oos-wndow
            target.CV.LambdaIdx = selections_CV_Lidx; % index of oos-window

            % Store the selected indices for AIC
            target.AIC.S = selections_AIC; % sparsify this // could use cell aswell with selections
            target.AIC.Lambda = selections_AIC_L; % Lasso/EN lamdas per oos-wndow
            target.AIC.LambdaIdx = selections_AIC_Lidx; % index of oos-window

            % Store the selected indices for BIC
            target.BIC.S = selections_BIC; % sparsify this // could use cell aswell with selections
            target.BIC.Lambda = selections_BIC_L; % Lasso/EN lamdas per oos-wndow
            target.BIC.LambdaIdx = selections_BIC_Lidx; % index of oos-window
            vs.(y_string).(['h',num2str(h)]).(valid_m_name) = target;
        end
    end

    toc

end

% Save variable selection for each target variable and horizon
save(['selection', filesep, 'vs_',opt.start_date(end-1:end),'_', ...
    char(opt.transformation_method),'_h_',num2str(h),'_', ...
    num2str(alpha_idx),'_',num2str(opt.target_id), '.mat'], 'opt', 'vs', 'data');

end