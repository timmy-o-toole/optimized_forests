function [qdates, unique_gn, reordered_grouped_smat, maxcols] = vs_group_results(vsic, h, vnt, vs, ds, strt, group_by, normalize_columns)
% This function loads and processes variable selection results for visualisation in a heatmap. 
% The results are grouped and optionally normalized.
%
% Inputs:
% vsic: String indicating the type of information criterion ('AIC', 'BIC' or 'CV').
% h: Integer indicating the horizon.
% vnt: String containing the variable name.
% vs: String indicating the variable selection method ('Lasso' or 'EN').
% ds: String indicating the dataset ('adaptive_fred', 'adaptive_fred_all', 'fred', 'fred_all').
% strt: String indicating the start date (e.g., '75' for 1975).
% group_by: String specifying the variable grouping method ('Transformation', 'Variable_Name', 'Variable_Group').
% normalize_columns: Boolean indicating whether to normalize columns before plotting.
%
% Outputs:
% qdates: Quarterly dates.
% unique_gn: Unique group names.
% reordered_grouped_smat: Processed sparse matrix.
% maxcols: Number of columns in the original sparse matrix.
%
% Note: This function is designed to work under specific conditions:
% The end date of the data is fixed to '01-12-2022' and 
% the out-of-sample length is constant with 192 observations.
% Changes to these settings may require adjustments to the function.

% Load handmade Mapping 
filename = 'fred_md_mapping.csv';
datamap = readtable(fullfile(pwd,'data', filename));
bcol = 2; 
rcol = 3; % Selected mapping cols from datamap
mapping = containers.Map(table2cell(datamap(:, 2)), table2cell(datamap(:, 3)));
vnorg = datamap.VAR;

% Date variable
end_date = datetime('01-12-2022', 'InputFormat', 'dd-MM-yyyy');
num_obs = 192; %size(grouped_smat, 1);
dates = end_date - calmonths(0:num_obs-1);
dates = flip(dates);

% Get the full path to the mat file
mat_file_path = fullfile(pwd, 'selection', ['vs_', strt, '_', char(ds), '.mat']);

% Load data selection and do data prepping
selection = load(mat_file_path, 'vs');
vnames = load(mat_file_path, 'data').data.series;

% Active Set
smat = selection.vs.(vnt{1}).(['h',num2str(h)]).(vs).(vsic).S;
maxcols =  size(smat, 2);

% Group 
switch group_by
    % Group By Functions

    case 'Transformation'
        % TRANSFORMATION [LOG...]
        grouped_varnames = cellfun(@(x) regexprep(x, '^[^_]*_', ''), vnames, 'UniformOutput', false);
        group_indices = findgroups(grouped_varnames);
        
    case 'Variable_Group'
        % VARIABLE NAME [CPI, INDPRO..]
        grouped_varnames = cellfun(@(x) regexp(x, '[^_]*', 'match', 'once'), vnames, 'UniformOutput', false);
        grouped_varnames = cellfun(@(x) mapping(x), grouped_varnames, 'UniformOutput', false);
        group_indices = findgroups(grouped_varnames);
        
    otherwise
        % VARIABLE GROUP [FINANCE, LABOUR...]
        grouped_varnames = cellfun(@(x) regexp(x, '[^_]*', 'match', 'once'), vnames, 'UniformOutput', false);
        group_indices = findgroups(grouped_varnames);
        
end

% Group by Category and process sparse matrix
group_indices = findgroups(grouped_varnames); 
grouped_smat = cell2mat(splitapply(@(x) {sum(x, 2)}, smat, group_indices));

% Reorder Variable Name
if strcmp(group_by, 'Variable_Name')
    is_element_in_grouped_varnames = ismember(vnorg, unique(grouped_varnames));
    unique_groups = vnorg(is_element_in_grouped_varnames);
    vnorg_group_indices = findgroups(unique_groups);
    reordered_grouped_smat =  grouped_smat(:, vnorg_group_indices);
else
    unique_groups = unique(grouped_varnames);
    reordered_grouped_smat = grouped_smat;
end

% Date to Quarterly
tData = array2timetable(reordered_grouped_smat,'RowTimes',dates');
tData = retime(tData, 'quarterly', 'sum');
reordered_grouped_smat = table2array(tData);
qdates = tData.Time;

% Normalize 
if normalize_columns
    reordered_grouped_smat = normalize_data(tData);
end

% Colnames: Add Number to Name
%rank_cs = round(tiedrank(sum(reordered_grouped_smat)'));
%unique_gn =  arrayfun(@(x,y) sprintf('%s (%d)', unique_groups{x}, y), (1:numel(unique_groups))', rank_cs, 'UniformOutput', false);
unique_gn = unique_groups;

end

function normalized_data = normalize_data(Data)
    % Check input data type and convert to array if necessary
    if istable(Data) || istimetable(Data)
        nData = table2array(Data);
    elseif isnumeric(Data) || islogical(Data)
        nData = Data;
    else
        error('Unsupported data type. Input data must be numeric, logical, table, or timetable.');
    end

    % Normalize the data
    min_values = min(nData, [], 1);
    max_values = max(nData, [], 1);
    range_values = max_values - min_values;
    normalized_data = (nData - min_values) ./ range_values;
end

