function [new_target_var, new_target_var_tot] = replace_var(target_var, data, opt)
% =========================================================================
% DESCRIPTION:
%           The code transforms the names of the given target variable.
%           Problem: Due to transformations, the name of the target var
%           is changed. So the original transformations is needed. The code
%           also returns a vector for all variables if "all" is included.
%
% -------------------------------------------------------------------------
% INPUT:
%           target_var   = string/s of fred_md target variable
%           csv_in = Recommended fred transformations. Fulfills
%           stationarity.
%
% OUTPUT:
%           new_target_var  = transformations_name and "all"
%           new_target_var_tot = variable names transformations for all if "all" include
%
% =========================================================================


if opt.run_pretransformation == 1

    % Load the original data
    data = data_transform(data, "fred");

    % Remove Outliers
    data = outlier_remove(data);

    % Remove NAs / Interpolation
    data = na_remove(data, opt);

    % Perform stationarity test
    data = stationarity_test(data, opt);

    % Create dataframe with two columns
    col1 = cellfun(@(x) strtok(x, '_'), data.series, 'UniformOutput', false);
    col2 = data.series;
    dum = table(col1', col2');
    dum.Properties.VariableNames = {'org_name', 'trans_name'};
    save('data_dict.mat', 'dum');

else
    % File name of desired FRED-MD vintage
    csv_in='data_dict.mat';

    % Load data from CSV file
    load(csv_in);
end

% Initialize new_target_var variable
new_target_var = {};

% Loop through each element of the target_var
for ii = 1:length(target_var)
    % Check if the element is "all"
    if strcmp(target_var{ii},'all')
        % Add "all" to the new_target_var
        new_target_var{end+1} = 'all';
    else
        % Find the index of the element in the org_name column
        index = find(strcmp(dum.org_name, target_var{ii}));
        % Check if the element is not found in the org_name column
        if isempty(index)
            % Raise an error with the element name
            error(['target_var ' target_var{ii} ' does not fulfill stationarity condition!']);
        else
            % Add the corresponding name in the trans_name column to the new_target_var
            new_target_var{end+1} = dum.trans_name{index};
        end
    end
end
% check if target_var includes 'all'
if any(strcmp(new_target_var,'all'))
    new_target_var_tot = dum.trans_name;
else
    new_target_var_tot = new_target_var;
end
end