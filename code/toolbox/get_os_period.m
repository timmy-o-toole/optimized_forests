% Function for determining the out-of-sample period
function [str_os_s, str_os_e, os_period] = get_os_period(dates, T, m)

os_period = dates(T-m+1:end, 1);
str_os_s = os_period{1}; str_os_e = os_period{end};

end