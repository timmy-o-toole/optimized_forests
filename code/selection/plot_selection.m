clear; clc;
addpath(genpath('..\toolbox\'));

load('vs_75_adaptive_fred.mat', 'data');
s_names_fred = data.series;
clear data;

transformation_set = 'fred_all';

if strcmp(transformation_set,'fred') || strcmp(transformation_set,'fred_all')
    load("customcolormap_fred_all.mat");
else
    load("customcolormap");
end
% load variable selection for specific dataset
load(['vs_75_groupwise_',transformation_set,'.mat']);

[T, N] = size(data.data_trans_outlier_and_na_removed_stationary);
[~, ~, os_period] = get_os_period(data.dates, T, opt.m);
s_names = data.series;
var_names = strtok(s_names, '_');
clear data;

idx_vars = [];
str_vars = [];
idx_fred_vars = [];
for ii = 1:length(s_names_fred)
    y_string = char(strtok(s_names_fred{ii}, '_'));
    list_str_vars = find(strcmp(var_names, y_string));
    idx_vars = [idx_vars, list_str_vars];
    str_vars = [str_vars, repmat(string(y_string), 1, length(list_str_vars))];
    idx_fred_vars = [idx_fred_vars, ones(1, length(list_str_vars))*ii];
end

% Specify h, target, and method
h = 1; target = 'CPIAUCSL'; method = 'EN'; reg_sel = 'CV'; 

sel_mat = vs.(target).(['h',num2str(h)]).(method).(reg_sel).S;

sel_mat_heat = zeros(size(sel_mat,1), length(s_names_fred));

for ii = 1:size(sel_mat,1)

    for jj = 1:length(s_names_fred)

        y_string_fred = char(strtok(s_names_fred{jj}, '_'));
        idx_fred_all = find(strcmp(var_names, y_string_fred));

        idx_active = find(sel_mat(ii, idx_fred_all));

        if ~isempty(idx_active)

            switch extractAfter(s_names{idx_fred_all(idx_active)},strtok(s_names{idx_fred_all(idx_active)}, '_'))
                case '_lvl'
                    sel_mat_heat(ii, jj) = 1;
                case '_dif_1m'
                    sel_mat_heat(ii, jj) = 2;
                case '_2nd_dif_1m'
                    sel_mat_heat(ii, jj) = 3;
                case '_log'
                    sel_mat_heat(ii, jj) = 4;
                case '_dif_log'
                    sel_mat_heat(ii, jj) = 5;
                case '_2nd_dif_log'
                    sel_mat_heat(ii, jj) = 6;
                case '_dif_pct'
                    sel_mat_heat(ii, jj) = 7;
                case '_pct_1y'
                    sel_mat_heat(ii, jj) = 8;
                case '_dif_1y'
                    sel_mat_heat(ii, jj) = 9;
                case '_pct_1q'
                    sel_mat_heat(ii, jj) = 10;
                case '_dif_1q'
                    sel_mat_heat(ii, jj) = 11;
                case '_box'
                    sel_mat_heat(ii, jj) = 12;
                case '_lvl_detrended'
                    sel_mat_heat(ii, jj) = 13;
                case '_dif_1m_detrended'
                    sel_mat_heat(ii, jj) = 14;
                case '_2nd_dif_1m_detrended'
                    sel_mat_heat(ii, jj) = 15;
                case '_log_detrended'
                    sel_mat_heat(ii, jj) = 16;
                case '_dif_log_detrended'
                    sel_mat_heat(ii, jj) = 17;
                case '_2nd_dif_log_detrended'
                    sel_mat_heat(ii, jj) = 18;
                case '_dif_pct_detrended'
                    sel_mat_heat(ii, jj) = 19;
                case '_pct_1y_detrended'
                    sel_mat_heat(ii, jj) = 20;
                case '_dif_1y_detrended'
                    sel_mat_heat(ii, jj) = 21;
                case '_pct_1q_detrended'
                    sel_mat_heat(ii, jj) = 22;
                case '_dif_1q_detrended'
                    sel_mat_heat(ii, jj) = 23;
                case '_box_detrended'
                    sel_mat_heat(ii, jj) = 24;
            end
        end
    end

end

figure()
hp = heatmap(sel_mat_heat, 'Colormap', CustomColormap);

YLabels = os_period;
idxY = 1:length(YLabels);
customYLabels = YLabels;

% Replace all but the fifth elements by spaces
for ii = 1:length(customYLabels)
    if mod(idxY(ii),5) ~= 0
        customYLabels{ii} = ' ';
    end
end
% Set the 'YDisplayLabels' property of the heatmap 
hp.YDisplayLabels = customYLabels;

