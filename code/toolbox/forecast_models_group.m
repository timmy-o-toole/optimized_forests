function results_all = forecast_models(data, opt)

% Settings
[T, N] = size(data.data_trans_outlier_and_na_removed_stationary);

% Calculate the out-of-sample period
[str_os_s, str_os_e] = get_os_period(data.dates, T, opt.m);
fprintf('Out-of-sample period: %s - %s\n', str_os_s, str_os_e);


% Define Target Variables
for fh = 1:length(opt.h)
    
    if(sum(strcmp(opt.vn, "all")) >= 1)
        
        ind_f_vars = 1:N;
        
    else
        
        % Get index of variables to forecast
        ind_f_vars = cellfun(@(x)find(contains(data.series,x)),opt.vnt);
        
    end
    
        
    % Define the target models in this list and order
    if opt.preselection == 1
        models = {'arp_f_ps_rf_recursive', 'rf_ps_rf_recursive', ...
            'arp_f_ps_groupwise_rf', 'rf_ps_groupwise_rf', ...
            'arp_f_ps_groupwise', 'rf_ps_groupwise', 'arp_f_ps', 'rf_ps', 'arp'};
    else 
        models = {'lasso', 'en', 'ada_lasso', 'lasso_aic', 'en_aic', 'ada_lasso_aic', ...
            'lasso_bic', 'en_bic', 'ada_lasso_bic', 'ridge', 'arp_f', 'rf', 'rw', 'arp'};
    end
    
    
    for i = 1:length(models)
        switch models{i}
            case 'rf'
                results = forecast_rf(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), data.series,4,4, opt);
            case 'lasso'
                results = forecast_vs(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'en'
                results = forecast_vs(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 0.5, opt);
            case 'ada_lasso'
                results = forecast_vs_adaptive(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'lasso_aic'
                results = forecast_vs_aic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'en_aic'
                results = forecast_vs_aic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 0.5, opt);
            case 'ada_lasso_aic'
                results = forecast_vs_adaptive_aic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'lasso_bic'
                results = forecast_vs_bic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'en_bic'
                results = forecast_vs_bic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 0.5, opt);
            case 'ada_lasso_bic'
                results = forecast_vs_adaptive_bic(data.data_trans_outlier_and_na_removed_stationary, data.series, ind_f_vars, opt.m, opt.h(fh), 1, opt);
            case 'ridge'
                results = forecast_ridge(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), opt);
            case 'arp_f'
                results = forecast_arp_f(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.nf_static, opt.c, opt.m, opt.h(fh), opt.max_AR, opt.max_F, opt.ic, opt.direct, opt);
            case 'rw'
                results = forecast_rw(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh));
            case 'arp'
                results = forecast_arp(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.c, opt.max_AR, opt.m, opt.h(fh), opt.ic, opt.direct);
            case 'arp_f_ps'
                results = forecast_arp_f_ps(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.nf_static, opt.c, opt.m, opt.h(fh), opt.max_AR, opt.max_F, opt.ic, opt.direct, opt);
            case 'rf_ps'
                results = forecast_rf_ps(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), data.series,4,4, opt);
            case 'arp_f_ps_groupwise'
                results = forecast_arp_f_ps_groupwise(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.nf_static, opt.c, opt.m, opt.h(fh), opt.max_AR, opt.max_F, opt.ic, opt.direct, opt);
            case 'rf_ps_groupwise'
                results = forecast_rf_ps_groupwise(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), data.series,4,4, opt);
            case 'arp_f_ps_groupwise_rf'
                results = forecast_arp_f_ps_groupwise_rf(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.nf_static, opt.c, opt.m, opt.h(fh), opt.max_AR, opt.max_F, opt.ic, opt.direct, opt);
            case 'rf_ps_groupwise_rf'
                results = forecast_rf_ps_groupwise_rf(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), data.series,4,4, opt);
            case 'arp_f_ps_rf_recursive'
                results = forecast_arp_f_ps_rf_recursive(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.nf_static, opt.c, opt.m, opt.h(fh), opt.max_AR, opt.max_F, opt.ic, opt.direct, opt);
            case 'rf_ps_rf_recursive'
                results = forecast_rf_ps_rf_recursive(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.m, opt.h(fh), data.series,4,4, opt);
        end
        
        % Store output in loop
        variable_name = join([opt.transformation_method, "__", opt.start_date(end-1:end), "_", opt.adaptive, ...
            "_", models{i}, "__", "h_", num2str(opt.h(fh)), "__", join(string(opt.target_id), "_")], "");
        save(fullfile("out",[variable_name + ".mat"]), "results", "opt");
        
        % Store results
        if ismember(models{i}, {'arp_f', 'arp_f_ps'})
            results_all.([models{i}, '_bn1']).(['h', num2str(opt.h(fh))]).results = results.results_bn1;
            results_all.([models{i}, '_ed']).(['h', num2str(opt.h(fh))]).results = results.results_ed;
            results_all.([models{i}, '_ts_bn1']).(['h', num2str(opt.h(fh))]).results = results.results_ts_bn1;
            results_all.([models{i}, '_em_bn1']).(['h', num2str(opt.h(fh))]).results = results.results_em_bn1;
            results_all.([models{i}, '_ts_ed']).(['h', num2str(opt.h(fh))]).results = results.results_ts_ed;
            results_all.([models{i}, '_em_ed']).(['h', num2str(opt.h(fh))]).results = results.results_em_ed;
            results_all.([models{i}, '_ts_saf']).(['h', num2str(opt.h(fh))]).results = results.results_ts_saf;
            results_all.([models{i}, '_em_saf']).(['h', num2str(opt.h(fh))]).results = results.results_em_saf;
            results_all.([models{i}, '_saf1']).(['h', num2str(opt.h(fh))]).results = results.results_saf1;
            results_all.([models{i}, '_saf2']).(['h', num2str(opt.h(fh))]).results = results.results_saf2;
            results_all.([models{i}, '_saf3']).(['h', num2str(opt.h(fh))]).results = results.results_saf3;
            results_all.([models{i}, '_saf4']).(['h', num2str(opt.h(fh))]).results = results.results_saf4;
        else
            results_all.(models{i}).(['h', num2str(opt.h(fh))]) = results;
        end
        
        
        
    end
    
end