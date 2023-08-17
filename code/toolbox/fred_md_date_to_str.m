% Function for transforming fred md dates into date string

function [str_date] = fred_md_date_to_str(dates)

str_date = cell(length(dates), 1);

for id_d = 1:length(dates)
    date = dates(id_d);

    reminder = date - floor(date);
    if (reminder == 0)
        mon = 12;
        year = date - 1;
    else
        mon = reminder * 12;
        year = floor(date);
    end

    str_date{id_d} = ['01.', sprintf('%02d.',round(mon)), sprintf('%d',year)];

end