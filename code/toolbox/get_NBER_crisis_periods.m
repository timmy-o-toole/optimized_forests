function crisis_months = get_NBER_crisis_periods(file_path,pre_crisi_months ,after_crisis_months)
% Get the dates from the excel file for recessions in the US

%% Get recession dates NBER

% Load the recession dates
recession_tab = readtable(file_path, 'ReadVariableNames', false);
recession_dates = recession_tab(2:35,:);
dates1 = datetime(string(datetime(regexprep(recession_dates.Var1, '\s*\([^)]*\)\s*', ''), 'InputFormat', 'MMMM yyyy'), 'dd.MM.yyyy') , 'InputFormat', 'dd.MM.yyyy');
dates2 = datetime(string(datetime(regexprep(recession_dates.Var2, '\s*\([^)]*\)\s*', ''), 'InputFormat', 'MMMM yyyy'), 'dd.MM.yyyy') , 'InputFormat', 'dd.MM.yyyy');

crisis_months = [];
% Loop over each row in the table
for i = 1:size(dates1, 1)
    % Generate monthly sequence of dates between the two dates in this row
    seq = dateshift(dates1(i), 'start', 'month', 0):calmonths(1):dateshift(dates2(i), 'start', 'month', 0);
    seq_start = dateshift(seq(1), 'start', 'month', -pre_crisi_months); % Shift the start date of the sequence by -3 months
    seq_end = dateshift(seq(end), 'start', 'month', after_crisis_months); % Shift the end date of the sequence by 24 months  
    new_seq = seq_start:calmonths(1):seq_end;
    crisis_months = [crisis_months, new_seq];    
end
crisis_months = unique(crisis_months);

end