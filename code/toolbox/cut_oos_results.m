function [data, os_date_vec] = cut_oos_results(data, start_date, end_date, m, os_start, os_end, crisis_months, sample_name)

% Get sequence of os_start to os_end date
os_start_dat = datetime(os_start, 'InputFormat', 'dd.MM.yyyy');
os_end_dat = datetime(os_end, 'InputFormat', 'dd.MM.yyyy');
os_seq = dateshift(os_start_dat, 'start', 'month', 0):calmonths(1):dateshift(os_end_dat, 'start', 'month', 0);

% Get sequence of start to end date
start_dat = datetime(start_date, 'InputFormat', 'dd.MM.yyyy');
end_dat = datetime(end_date, 'InputFormat', 'dd.MM.yyyy');
seq = dateshift(start_dat, 'start', 'month', 0):calmonths(1):dateshift(end_dat, 'start', 'month', 0);

% Get out-of-sample start and end dates
[~, T] = size(seq);
os_period = seq(1,T-m+1:end);

% Find indices for out-of-sample start and end date from dates
[common_values, ~, idx_os_period] = intersect(os_seq, os_period);

%fprintf('%s %s %s %s - %s\n', 'Sample:',string(sample_name),', Out-of-sample period:' , string(common_values(1)), string(common_values(end)));

% Cut errors according to end and start date for out-of-sample window
data = data(idx_os_period);
os_date_vec = common_values;

% Cut Sample according to crisis and non-crisis period definition
switch string(sample_name)
    case 'full'
    % Do nothing
    case'crisis'
        [common_values_crisis, ~, idx_os_period_crisis] = intersect(crisis_months, common_values);
        data = data(idx_os_period_crisis);
        os_date_vec = common_values_crisis;
    case'non-crisis'
        idx_os_period_non_crisis = find(ismember(common_values, setdiff(common_values, crisis_months)));
        common_values_non_crisis = common_values(idx_os_period_non_crisis);
        data = data(idx_os_period_non_crisis);
        os_date_vec = common_values_non_crisis;
end


end
