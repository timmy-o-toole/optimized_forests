function data = load_data(csv_name)

% File name of desired FRED-MD vintage
csv_in=csv_name;

% Load data from CSV file
dum=importdata(csv_in,',');

% Variable names
series=dum.textdata(1,2:end);
% Replace remove white spaces in variable names
series=strrep(series,' ','');
series=strrep(series,'&','');
series=strrep(series,':','');

% Transformation numbers
tcode=dum.data(1,:);
tcode(tcode == 4) = 5; % Take first difference of housing variables for stationarity 
tcode(tcode == 6) = 5;
% Take second difference of the following variables for stationarity
tcode(strcmp(series, 'CPIMEDSL')) = 6;
tcode(strcmp(series, 'DDURRG3M086SBEA')) = 6;
tcode(strcmp(series, 'DSERRG3M086SBEA')) = 6;
tcode(strcmp(series, 'CES0600000008')) = 6;


% Raw data
rawdata=dum.data(2:end,:);

% Month/year of final observation
final_datevec=datevec(dum.textdata(end,1));
final_month=final_datevec(2);
final_year=final_datevec(1);

% Dates (monthly) are of the form YEAR+MONTH/12
% e.g. March 1970 is represented as 1970+3/12
% Dates go from 1959:01 to final_year:final_month (see above)
dates = (1959+1/12:1/12:final_year+final_month/12)';

% T = number of months in sample
T=size(dates,1);
rawdata=rawdata(1:T,:);

data.data_raw = rawdata;
data.tcode = tcode;
data.dates = dates;
data.series = series;

end