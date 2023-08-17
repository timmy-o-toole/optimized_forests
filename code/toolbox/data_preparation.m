function[dta] = data_preparation(opt)

data_struc = load('data_fred.mat'); % load monthly data that is available from 1959M2-2021M4
% Get index for start dates and end dates
dates = fred_md_date_to_str(data_struc.dates);
start_d = find(strcmp(dates, opt.start_date)); 
end_d = find(strcmp(dates, opt.end_date));

% Interpolate missings using static factor model
% X = interpolate_missings(data(start_d:end_d, :));

% Select data range as set above and remove nan values
names = data_struc.series(~isnan(mean(data_struc.data(start_d:end_d,:))));
X = data_struc.data(start_d:end_d, ~isnan(mean(data_struc.data(start_d:end_d,:))));
dates = dates(start_d:end_d);

dta.data = X;
dta.names = names;
dta.dates = dates;
dta.T = size(X, 1);
dta.N = size(X, 2);

end