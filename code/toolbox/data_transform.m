function data = data_transform(data, opt)
% =========================================================================
% DESCRIPTION: 
% This function runs the selected transformation function
%
% -------------------------------------------------------------------------
% INPUT:
%           data     = raw data 
%           opt      = options
%
% OUTPUT: 
%           data     = transformed data
%
% =========================================================================

switch (opt)
    
    case 'fred'
data = data_transform_fred(data); % Perform Fred transformations

    case 'fred_all'
data = data_transform_all_fred(data); % Perform all Fred transformations on all variables

    case 'all'
data = data_transform_all(data); % Perform all transformations on all variables


end