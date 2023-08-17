function [tab_result, time_res, vars, os_date_vec] = evaluate_models(fc_results, data, os_start, os_end, crisis_months, sample_name)

%% Setup

% Cache series_names
series_names = data.series;

% Benchmark model
m_bench = "arp";


%% Start Evaluation Loop

% Get dataset names
d_names = fieldnames(fc_results);

for ds = 1:length(d_names)

    % Get model names
    m_names = fieldnames(fc_results.(d_names{ds}));
    
    % Remove benchmark from list
    m_names(strcmp(m_names, m_bench)) = [];

    for mn = 1:length(m_names)
    
        % Get forecasting step
        h_step = fieldnames(fc_results.(d_names{ds}).(m_names{mn}));
   
        for fh = 1:length(h_step)
         
            % Initialize structure for target variable names
            vars.(d_names{ds}).(m_names{mn}).(h_step{fh}) = struct();
            vars.(d_names{ds}).(m_names{mn}).(h_step{fh}).var_names = strings(1, 1);

            % Get target variable name
            tar_var = fieldnames(fc_results.(d_names{ds}).(m_names{mn}).(h_step{fh}));
            
            for tv = 1:length(tar_var)
                %% Get target variable name

                bm_tar_var_idx = find(strcmp(fieldnames(fc_results.(d_names{ds}).(m_bench).(h_step{fh})), tar_var{tv}));

                bm_tar_var_names = fieldnames(fc_results.(d_names{ds}).(m_bench).(h_step{fh}));

                % Get options from benchmark results and save variable name
                opt_bm = fc_results.(d_names{ds}).(m_bench).(h_step{fh}).(bm_tar_var_names{bm_tar_var_idx}).opt;
                vars.(d_names{ds}).(m_bench).(h_step{fh}).var_names(1,bm_tar_var_idx) = opt_bm.vn;

                % Get options from selected results and save variable name
                opt_mn = fc_results.(d_names{ds}).(m_names{mn}).(h_step{fh}).(tar_var{tv}).opt;
                vars.(d_names{ds}).(m_names{mn}).(h_step{fh}).var_names(1,tv) = opt_mn.vn;
                
                %% Evaluation
                
                % Get errors of benchmark model
                fe_bench = fc_results.(d_names{ds}).(m_bench).(h_step{fh}).(tar_var{tv}).results.err;

                % Cut errors according to end and start date for out-of-sample window
                [fe_bench, os_date_vec] = cut_oos_results(fe_bench, opt_bm.start_date, opt_bm.end_date, opt_bm.m, os_start, os_end, crisis_months, sample_name);

                % Save benchmark errors into structure
                time_res.(d_names{ds}).(h_step{fh}).(m_bench).time_errors(:,tv) = fe_bench;

                % Get index of selected target variable
                ind_f_vars = find(contains(series_names, opt_mn.vn));
                
                % Save mean squared error of benchmark model
                res.(d_names{ds}).(h_step{fh}).res_mat(1, (ind_f_vars-1)*2+1) = mean(fe_bench.^2);
                res.('all').(h_step{fh}).res_mat(1, (ind_f_vars-1)*2+1) = mean(fe_bench.^2);

                % Get errors of model
                fe_model = fc_results.(d_names{ds}).(m_names{mn}).(h_step{fh}).(tar_var{tv}).results.err;
    
                % Cut errors according to end and start date for out-of-sample window
                [fe_model, os_date_vec] = cut_oos_results(fe_model, opt_mn.start_date, opt_mn.end_date, opt_mn.m, os_start, os_end, crisis_months, sample_name);

                % Save model errors into structure
                time_res.(d_names{ds}).(h_step{fh}).(m_names{mn}).time_errors(:,tv) = fe_model; 

                % Calculate Diebold-Mariano test
                try
                [~, pval_dm] = dm_test(fe_model, fe_bench, opt_mn.h);
                catch
                    diff = sum(fe_model - fe_model);
                    fprintf('%s %d %s %s %s %s\n', 'sum of difference between fe_model and fe_bench is:',diff, d_names{ds}, h_step{fh}, m_names{mn}, tar_var{tv});
                    pval_dm = 999;
                end 
                % Save mean squared error
                res.(d_names{ds}).(h_step{fh}).res_mat(mn+1, (ind_f_vars-1)*2+1) = mean(fe_model.^2); 
                res.('all').(h_step{fh}).res_mat(1+mn+length(m_names)*(ds-1), (ind_f_vars-1)*2+1) = mean(fe_model.^2);

                % Save relative mean squared error
                res.(d_names{ds}).(h_step{fh}).res_mat_rel(mn, (ind_f_vars-1)*2+1) = mean(fe_model.^2) ./ mean(fe_bench.^2);
                res.('all').(h_step{fh}).res_mat_rel(mn+length(m_names)*(ds-1), (ind_f_vars-1)*2+1) = mean(fe_model.^2) ./ mean(fe_bench.^2);

                % Save pvalue of DM-test in res_mat
                res.(d_names{ds}).(h_step{fh}).res_mat(mn+1, ind_f_vars*2) = pval_dm;
                res.('all').(h_step{fh}).res_mat(1+mn+length(m_names)*(ds-1), ind_f_vars*2) = pval_dm;

                % Save pvalue of DM-test in res_mat_rel
                res.(d_names{ds}).(h_step{fh}).res_mat_rel(mn, ind_f_vars*2) = pval_dm;
                res.('all').(h_step{fh}).res_mat_rel(mn+length(m_names)*(ds-1), ind_f_vars*2) = pval_dm;                              
                
            end                
        end
    end 
end

%% Delete empty columns and add errors over all variables

% Cache data set names
d_names_plus = d_names;

% Delete empty columns
for ds = 1:length(d_names_plus)
    for fh = 1:length(h_step)
        
        % Find non zero columns
        non_zero_cols.(d_names_plus{ds}) = any(res.(d_names_plus{ds}).(h_step{fh}).res_mat,1);

        % Find zero columns in res_mat and delete those columns
        zero_cols = find(all(res.(d_names_plus{ds}).(h_step{fh}).res_mat == 0, 1));
        res.(d_names_plus{ds}).(h_step{fh}).res_mat(:,zero_cols) = [];

        % Find zero columns in res_mat_rel and delete those columns
        zero_cols = find(all(res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel == 0, 1));
        res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel(:,zero_cols) = [];  

        % Define two empty arrays for mean over all target variables
        z_cols_mat = zeros(size(res.(d_names_plus{ds}).(h_step{fh}).res_mat,1),2);
        z_cols_mat_rel = zeros(size(res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel,1),2);

        % Add two empty columns
        res.(d_names_plus{ds}).(h_step{fh}).res_mat = [res.(d_names_plus{ds}).(h_step{fh}).res_mat, z_cols_mat];
        res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel = [res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel, z_cols_mat_rel];

        if ds == 1 % Only add once two extra columns of zeros for all
                % Find non zero columns
                non_zero_cols.('all') = any(res.('all').(h_step{fh}).res_mat,1);
        
                % Find zero columns in res_mat and delete those columns
                zero_cols_all = find(all(res.('all').(h_step{fh}).res_mat == 0, 1));
                res.('all').(h_step{fh}).res_mat(:,zero_cols_all) = [];
        
                % Find zero columns in res_mat_rel and delete those columns
                zero_cols_all = find(all(res.('all').(h_step{fh}).res_mat_rel == 0, 1));
                res.('all').(h_step{fh}).res_mat_rel(:,zero_cols_all) = [];  
        
                % Define two empty arrays for mean over all target variables
                z_cols_mat_all = zeros(size(res.('all').(h_step{fh}).res_mat,1),2);
                z_cols_mat_rel_all = zeros(size(res.('all').(h_step{fh}).res_mat_rel,1),2);
        
                % Add two empty columns
                res.('all').(h_step{fh}).res_mat = [res.('all').(h_step{fh}).res_mat, z_cols_mat_all];
                res.('all').(h_step{fh}).res_mat_rel = [res.('all').(h_step{fh}).res_mat_rel, z_cols_mat_rel_all];
        end

        % Define structure specific model array and delete benchmark model from it
        m_names_new = fieldnames(time_res.(d_names_plus{ds}).(h_step{fh}));
        m_names_new(strcmp(m_names_new, m_bench)) = [];
    
            for mn = 1:length(m_names_new)
    
                % Get forecast horizon from selected results structure
                h_step_numbers = cellfun(@(x) str2double(regexp(x, '\d+', 'match')), h_step);
    
                % Get errors of mean over all target variables
                all_bench = mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_bench).time_errors,2);
                all_model = mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_names_new{mn}).time_errors,2);
    
                % Calculate Diebold-Mariano test
                try
                [~, pval_dm] = dm_test(all_model, all_bench, h_step_numbers(fh));
                catch
                    diff = sum(all_model - all_bench);
                    fprintf('%s %d %s %s %s %s\n', 'sum of difference between all_model and all_bench is:',diff, d_names{ds}, h_step{fh}, m_names{mn}, tar_var{tv});
                    pval_dm = 999;
                end 
                % Save mean squared error
                res.(d_names_plus{ds}).(h_step{fh}).res_mat(1, end-1) = mean(mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_bench).time_errors.^2));
                res.(d_names_plus{ds}).(h_step{fh}).res_mat(mn+1, end-1) = mean(mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_names_new{mn}).time_errors.^2));
    
                % Save relative mean squared error
                res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel(mn, end-1) = mean(all_model.^2) ./ mean(all_bench.^2);
    
                % Save pvalue of DM-test
                res.(d_names_plus{ds}).(h_step{fh}).res_mat(mn+1, end) = pval_dm;
                res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel(mn, end) = pval_dm;
                
                %% Save results for data set all
                % Save mean squared error
                res.('all').(h_step{fh}).res_mat(1, end-1) = mean(mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_bench).time_errors.^2));
                res.('all').(h_step{fh}).res_mat(1+mn+length(m_names)*(ds-1), end-1) = mean(mean(time_res.(d_names_plus{ds}).(h_step{fh}).(m_names_new{mn}).time_errors.^2)); 
    
                % Save relative mean squared error
                res.('all').(h_step{fh}).res_mat_rel(mn+length(m_names)*(ds-1), end-1) = mean(all_model.^2) ./ mean(all_bench.^2);
    
                % Save pvalue of DM-test
                res.('all').(h_step{fh}).res_mat(1+mn+length(m_names)*(ds-1), end) = pval_dm;
                res.('all').(h_step{fh}).res_mat_rel(mn+length(m_names)*(ds-1), end) = pval_dm;            
            end
    end
end

%% Create Output Tables

% Add 'all' to data set name
d_names_plus = [d_names; {'all'}];

for ds = 1:length(d_names_plus)
    for fh = 1:length(h_step)
        
        % Initialize var_names
        var_names = cell(1,(length(series_names)+1)*2);
        
        % Add all to series names
        series_names = [series_names, 'all'];
        
        % Fill column and row names for tables
        for i = 1:length(var_names)
            if mod(i, 2) == 1 % odd position
            var_names(i) = {series_names{(i+1)/2}};
            else % even position
            var_names(i) = {['pv_',series_names{i/2}]};
            end
        end
        
        % Select variable names that have non zero entries
        non_zero_cols_idx = [find(non_zero_cols.(d_names_plus{ds}) ==1), size(var_names,2)-1, size(var_names,2)]; % Here are two columns added for all
        var_names = var_names(non_zero_cols_idx);
         
        % Define the model names including the data set name for structur all
        if strcmp(d_names_plus{ds},'all')
            m_names_new = [];
            for dd = 1:length(d_names)
                set_names = fieldnames(time_res.(d_names{dd}).(h_step{fh}));
                set_names(strcmp(set_names, m_bench)) = [];
                set_names_ds = cellfun(@(x) [x, strcat('_',d_names{dd})], set_names, 'UniformOutput', false);
                m_names_new = [m_names_new; set_names_ds];
            end
        else

        % Define structure specific model array and delete benchmark model from it
        m_names_new = fieldnames(time_res.(d_names_plus{ds}).(h_step{fh}));
        m_names_new(strcmp(m_names_new, m_bench)) = [];
        end

        % Define format as strings
        row_names_t_mse = cellstr([m_bench; m_names_new]);
        row_names_t_rel = cellstr(m_names_new);

        % Delete empty rows with come from missing models
        zero_idx = find(all(res.(d_names_plus{ds}).(h_step{fh}).res_mat==0, 2));
        res.(d_names_plus{ds}).(h_step{fh}).res_mat(zero_idx,:) = [];
        zero_idx = find(all(res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel==0, 2));
        res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel(zero_idx,:) = [];

        % Create output table
        tab_result.(d_names_plus{ds}).(h_step{fh}).t_mse = array2table(res.(d_names_plus{ds}).(h_step{fh}).res_mat,'RowNames', transpose(row_names_t_mse),'VariableNames', var_names);
        tab_result.(d_names_plus{ds}).(h_step{fh}).t_rel = array2table(res.(d_names_plus{ds}).(h_step{fh}).res_mat_rel,'VariableNames', var_names, 'RowNames', row_names_t_rel);

    end
end

end

