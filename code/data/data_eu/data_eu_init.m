clear; clc;

data_eu = readtable("data_eu.csv","TreatAsMissing", "NA");
keys_title = readtable("keys_titles.csv");

data_eu_nd = data_eu(:, 2:end);

start_d = 241;
end_d = 571;

data = data_eu_nd{start_d:end_d, ~isnan(data_eu{start_d, 2:end})};
keys_t = keys_title{~isnan(data_eu{start_d, 2:end}), :};

data_t = data(:, ~isnan(mean(data)));
keys_t = keys_t(~isnan(mean(data)), :);

n_countries = {'ea', 'be', 'de', 'fra', 'it', 'es', 'nl'};
vars_groups = cell(2, length(n_countries));
vars_groups(1,:) = n_countries;
for ii = 1:length(n_countries)

    vars_groups(2,ii) = {sum(strcmp(keys_t(:,3), n_countries(ii)))};

end

var_names = cell(1, length(keys_t(:,1)));
for ii = 1:length(keys_t(:,1))
    var_names(1, ii) = keys_t(ii,1);
end

data_eu_table = [table(data_eu{start_d:end_d, 1}, 'VariableNames', {'Date'}), ...
    array2table(data_t, 'VariableNames', var_names)];
data_eu_keys_title_table = array2table(keys_t(:, 2:end)', 'VariableNames', var_names);

writetable(data_eu_table, 'data_filtered/data_eu_final.csv', 'Delimiter',';');
writetable(data_eu_keys_title_table, 'data_filtered/data_eu_keys_title_final.csv', 'Delimiter',';');
