function results = forecast_models(data, opt)

[T, N] = size(data.data_trans_outlier_and_na_removed_stationary);
% Calculate the out-of-sample period
[str_os_s, str_os_e] = get_os_period(data.dates, T, opt.m);
fprintf('Out-of-sample period: %s - %s\n', str_os_s, str_os_e);


for fh = 1:length(opt.h)
    
    if(sum(strcmp(opt.vn, "all")) >= 1)
        
        ind_f_vars = 1:N;
        
    else
        
        % Get index of variables to forecast
        ind_f_vars = find(ismember(data.series, opt.vnt));
        
    end
    
    for i = 0:4
        tic;
        switch i
            case 0
                f_vars = [1,1];
            case 1
                f_vars = [0,1,2,3,4,5,6,7,8,9,10];
            case 2
                f_vars = [1,2,4,5];
            case 3
                f_vars = [1,2];
            case 4
                f_vars = [1:50];
        end
        variable_name = ['results_rf_class_', num2str(i), '_h_', num2str(fh)];
        variable_value = forecast_rf_classes(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.c, opt.max_AR, opt.m, opt.ic, opt.h(fh), data.series, 5, 5, f_vars);
        save([variable_name, '.mat'], 'variable_value');
        
        % Load the saved result and save it to the `results` struct
        results.(['rf', num2str(i)]).(['h', num2str(opt.h(fh))]) = variable_value;
        fprintf('%s model completed\n', f_vars);
        toc
    end
    
    % AR(p) forecast
    results_arp = forecast_arp(data.data_trans_outlier_and_na_removed_stationary, ind_f_vars, opt.c, opt.max_AR, opt.m, opt.h(fh), opt.ic, opt.direct);
    
    results.arp.(['h', num2str(opt.h(fh))]) = results_arp;
    %     results.bn1.(['h', num2str(opt.h(fh))]) = results_arp_f.results_bn1;
    %     results.bn2.(['h', num2str(opt.h(fh))]) = results_arp_f.results_bn2;
    %     results.ed.(['h', num2str(opt.h(fh))]) = results_arp_f.results_ed;
    %     results.saf1.(['h', num2str(opt.h(fh))]) = results_arp_f.results_saf1;
    %     results.saf2.(['h', num2str(opt.h(fh))]) = results_arp_f.results_saf2;
    %     results.saf3.(['h', num2str(opt.h(fh))]) = results_arp_f.results_saf3;
    %     results.saf4.(['h', num2str(opt.h(fh))]) = results_arp_f.results_saf4;
    
end


end