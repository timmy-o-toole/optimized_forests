clear; clc;
addpath([genpath(['data', filesep]), genpath(['models', filesep]), genpath(['toolbox', filesep])]);

% Load settings
options

% Set horizon [1, 3, 6, 12]
horizon = 1

% Assuming opt.vn contains the variable names
min_ones = zeros(1, length(variable_names));

% Loop through all variable names
for ii = 1:length(variable_names)
    
    y_string_trans = opt.vn(ii); %'INDPRO_dif_log'
    y_string = char(strtok(y_string_trans, '_')); %"INDPRO"
    subset_method = "EN"; % "Lasso" or "EN"
    ic_VS = "AIC";
    
    % Load the corresponding selection matrix
    smat = load(['selection', filesep, 'vs_', char(opt.transformation_method), '.mat'], 'vs').vs.(y_string).(['h',num2str(horizon)]).(subset_method).(ic_VS).S;
    
    opt.vn(ii)
    
    % Count the number of ones in each row and find the minimum
    row_sums = sum(smat, 2);
    min_ones(ii) = min(row_sums);
end

% Print the minimum number of ones for each row in all selection matrices
disp(min_ones);

% display names to drop
variable_names(min_ones < 2)

