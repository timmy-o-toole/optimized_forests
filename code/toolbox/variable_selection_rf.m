function vs = variable_selection_rf(data, opt)
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
% opt.vntt = string(name + "+" + transformation)
target_vars = opt.vntt;
n_targets = length(target_vars); % Number of targets
m = opt.m; % Number of out-of-sample periods
num_trees = 500; % Medeiros default in R package
tic

% (start 1) Selection Method=====================
% Is defined

for idx_h = 1:length(opt.h)
    h = opt.h(idx_h);

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
        selections_CV = zeros(m, size(X,2));

        % Run Selection for each out-of-sample window
        % Use parfor instead of for
        for tt = 1:m
            tic
            % subset and standardize
            Xsub = X(tt:T-m+tt-h-1,:);
            ysub = y(tt+h:T-m+tt-1);

            Xn = zscore(Xsub);
            yn = zscore(ysub);

            % IDENTIFY GROUPS
            % SELECT WHERE SPARSITY OF GROUP IS ONE
            % PROBLEM: WHAT IF MORE THAN ONE IS SELECTED

            % Get variable names
            vnames = data.series;
            Xn_est = Xn;
            vnames_est = vnames;
            selections_CV_group = false(1, size(Xn,2));
            while ~isempty(Xn_est)

                % Compute Random Forest
                rf = TreeBagger(num_trees,Xn_est, yn, 'Method','regression', 'OOBPredictorImportance', 'on',...
                    'PredictorSelection','curvature','Prior','Uniform');
                
                % Compute Random Forest
                rf_wr = TreeBagger(num_trees,Xn_est, yn, 'Method','regression', 'SampleWithReplacement', 'off', ...
                    'InBagFraction', 0.7, 'Prior', 'Uniform', 'OOBPredictorImportance', 'on');


                % Extract index of predictor with highest importance
                [~,max_idx_imp] = max(rf.OOBPermutedPredictorDeltaError);
                selections_CV_group(1, strcmp(vnames,string(vnames_est(max_idx_imp)))) = 1;

                group_idx = find(strcmp(string(strtok(vnames_est(max_idx_imp),'_')), strtok(vnames_est, '_') ));
                Xn_est = Xn_est(:, setdiff(1:size(Xn_est,2),group_idx));
                vnames_est = vnames_est(1,setdiff(1:size(vnames_est,2),group_idx));

                sum(selections_CV_group,2)
            end

            % Set the selections for this time point
            selections_CV(tt,:) = selections_CV_group;
        toc
        end

        % I do not store these as we look at it grouped wise

        % (End 3) Set Each Variable=================
        % Store the selected indices for CV
        target.CV.S = selections_CV; % sparsify this // could use cell aswell with selections
        vs.(y_string).(['h',num2str(h)]) = target;
    end
end
toc
% Save variable selection for each target variable and horizon
save(['selection3', filesep, 'vs_',opt.start_date(end-1:end),'_','rf_recursive','_' ,...
    char(opt.transformation_method),'_h_',num2str(h),'_', ...
    num2str(opt.target_id), '.mat'], 'opt', 'vs', 'data');

end