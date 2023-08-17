function plot_heatmap(lam_s, names, os_dates, os_period)

load Colormap_con map_con

N = size(lam_s{1}, 1);
var_label = [0,5,19,47,57,62,72,75,83,90,95,122];
label_cen = nan(length(var_label)-1, 1);
for ll = 1:length(var_label)-1
    label_cen(ll) = ceil((var_label(ll+1) - var_label(ll))/2) + var_label(ll);
end
y_names = {};
for ii = 1:N
    if(ii == label_cen(1))
        y_names = [y_names, 'Consumption'];
    elseif(ii == label_cen(2))
        y_names = [y_names, 'IP'];
    elseif (ii == label_cen(3))
        y_names = [y_names, 'Labor market'];
    elseif (ii == label_cen(4))
        y_names = [y_names, 'Housing'];
    elseif (ii == label_cen(5))
        y_names = [y_names, 'Orders'];
    elseif (ii == label_cen(6))
        y_names = [y_names, 'Money'];
    elseif (ii == label_cen(7))
        y_names = [y_names, 'Stock market'];
    elseif (ii == label_cen(8))
        y_names = [y_names, 'Interest rates'];      
    elseif (ii == label_cen(9))
        y_names = [y_names, 'Spreads'];  
    elseif (ii == label_cen(10))
        y_names = [y_names, 'Exchange rates'];    
    elseif (ii == label_cen(11))
        y_names = [y_names, 'Prices'];
    else
        y_names = [y_names, ' '];
    end
end

for dd = 1:length(os_dates)

    indx = get_index_date(os_dates(dd), os_period);

    lam_e = lam_s{indx};

    a = max(max(lam_e));
    b = min(min(lam_e));
    c = ceil(max([a, abs(b)]));


    %[ f_names, f_names_s ] = factor_names( lam_e, names );

    xvalues = 1:size(lam_e, 2); %f_names_s;

    figure
    heatmap1(lam_e, xvalues, y_names, [], 'Colorbar', true, ...
        'GridLines', '-', 'GridY', var_label, 'ShowAllTicks', 1, ...
        'Colormap', map_con, 'MinColorValue', -c , ...
        'MaxColorValue', c);
    title(['Loadings matrix for ', os_dates(dd)]);

    % set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 20 22]);
    % saveas(gcf,['figures\factor_plots\hm_lam_e', '.eps'],'epsc')
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to obtain the index of out-of-sample period based on date
% string
function indx = get_index_date(dates_str, os_period)

date_str = strsplit(dates_str, '/');
date_m = double(date_str(1));
date_y = double(date_str(2));

date_num = date_y + date_m / 12;

indx = find(abs(os_period-date_num) < 1e-8, 1);

end


% Function to match factor naming automatically
function [ f_names, f_names_s ] = factor_names( lam_e, names )
% This funciton is used to determine which factor is associated to which
% time series

boundaries = 0.2;

% indicator showing which time series load on which factor: based on some
% boundaries
indicator = (abs(lam_e)>= boundaries);

KK = size(lam_e,2);

f_names = {};
f_names_s = {};

for k = 1:KK

    
     if (sum(floor(median(find(indicator(:,k) == 1))) == find(...
             strcmp(names, 'hwi')):find(strcmp(names, 'hours'))) == 1)
        % employment and unemployment
        if(sum(strcmp(f_names, 'labor market')) >= 1)
            f_names = [f_names, ['labor market', num2str(k)]];
            f_names_s = [f_names_s, 'LM', num2str(k)];
        else
            f_names = [f_names, 'labor market'];
            f_names_s = [f_names_s, 'LM'];            
        end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'housing')):find(strcmp(names, ...
            'housing permits W'))) == 1)
        % housing
        if(sum(strcmp(f_names, 'housing')) >= 1)
            f_names = [f_names, ['housing', num2str(k)]];
            f_names_s = [f_names_s, 'H', num2str(k)];            
        else
            f_names = [f_names, 'housing'];
            f_names_s = [f_names_s, 'H'];            
        end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'PPI goods')):find(strcmp(names, ...
            'PCE durable'))) == 1)
        % price
        if(sum(strcmp(f_names, 'price')) >= 1)
            f_names = [f_names, ['price', num2str(k)]];
            f_names_s = [f_names_s, 'P', num2str(k)];            
        else
            f_names = [f_names, 'price'];
            f_names_s = [f_names_s, 'P'];            
        end
     elseif (sum(floor(min(find(indicator(:,k) == 1))) == find(...
             strcmp(names, 'MAAA')):find(strcmp(names, ...
             'M30Y'))) == 1)
         % excess returns
         if(sum(strcmp(f_names, 'credit spreads')) >= 1)
             f_names = [f_names, ['credit spreads', num2str(k)]];
             f_names_s = [f_names_s, 'CS', num2str(k)];
         else
             f_names = [f_names, 'credit spreads'];
             f_names_s = [f_names_s, 'CS'];
         end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'commercial paper rate')):find(strcmp(names, ...
            'BAA bond yield'))) == 1) 
        % interest rates
        if(sum(strcmp(f_names, 'interest rates')) >= 1)
            f_names = [f_names, ['interest rates', num2str(k)]];
            f_names_s = [f_names_s, 'IR', num2str(k)]; 
        else
            f_names = [f_names, 'interest rates'];
            f_names_s = [f_names_s, 'IR']; 
        end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'income')):find(strcmp(names, 'capacity'))) == 1)
        % IP
        if(sum(strcmp(f_names, 'industrial production')) >= 1)
            f_names = [f_names, ['industrial production', num2str(k)]];
            f_names_s = [f_names_s, 'IP', num2str(k)];            
        else
            f_names = [f_names, 'industrial production'];
            f_names_s = [f_names_s, 'IP'];            
        end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'S&P composite')):find(strcmp(names, ...
            'Nasdaq industrial'))) == 1)
        % stock market
        if(sum(strcmp(f_names, 'stock market')) >= 1)
            f_names = [f_names, ['stock market', num2str(k)]];
            f_names_s = [f_names_s, 'SM', num2str(k)];            
        else
            f_names = [f_names, 'stock market'];
            f_names_s = [f_names_s, 'SM'];            
        end
%     elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
%             strcmp(names, 'S&P composite')):find(strcmp(names, ...
%             'VXO'))) == 1)
%         % stock market
%         if(sum(strcmp(f_names, 'stock market')) >= 1)
%             f_names = [f_names, ['stock market', num2str(k)]];
%         else
%             f_names = [f_names, 'stock market'];
%         end
    elseif (sum(floor(median(find(indicator(:,k) == 1))) == find(...
            strcmp(names, 'temp')):find(strcmp(names, ...
            'consumer credit'))) == 1)
        % Money
        if(sum(strcmp(f_names, 'Money')) >= 1)
            f_names = [f_names, ['Money', num2str(k)]];
        else
            f_names = [f_names, 'Money'];
        end
     end  
    
end
end