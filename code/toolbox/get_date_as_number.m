function start_date_transform = get_date_as_number(date)
serial_date_number = datenum(date, 'dd.mm.yyyy');
start_of_year = datenum([year(serial_date_number), 1, 1]);
num_days_in_year = datenum([year(serial_date_number)+1, 1, 1]) - start_of_year;
start_date_transform = year(serial_date_number) + (serial_date_number - start_of_year)/num_days_in_year;
end