function data = data_transform_fred(data)
% =========================================================================
% DESCRIPTION:
% This function transforms raw data based on each series' transformation
% code.
%
% -------------------------------------------------------------------------
% INPUT:
%           data     = raw data
%
% OUTPUT:
%           data          = transformed data
%
% -------------------------------------------------------------------------
% SUBFUNCTION:
%           transxf:    transforms a single series as specified by a
%                       given transfromation code
%
% =========================================================================
% APPLY TRANSFORMATION:
% Initialize output variable
yt = [];
st = [];
dat = data.data_raw;
tcode = data.tcode;
series = data.series;

% Number of series kept
N = size(dat,2);

% Perform transformation using subfunction transxf (see below for details)
for i = 1:N
    [dum, new_ser] = transxf(dat(:,i),tcode(i), series(i));
    %[dum, new_ser] = transxf(dat(:,:), i, series);
    yt = [yt, dum];
    st = [st, new_ser];
end

% Delete series with only NAN values
idx_nan = all(isnan(yt));
yt = yt(:,~idx_nan); %cols that are all nan
st = st(:,~idx_nan);

% Save transformed data into structure
data.data_trans = yt;
data.series = st;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION

function [y, new_series]=transxf(x, tcode, series)
% =========================================================================
% DESCRIPTION:
% This function transforms a single series (in a column vector)as specified
% by a given transfromation code.
%
% -------------------------------------------------------------------------
% INPUT:
%           x       = series (in a column vector) to be transformed
%           tcode   = transformation code (1-7)
%
% OUTPUT:
%           y       = transformed series (as a column vector)
%
% =========================================================================
% SETUP:
% Number of observations (including missing values)
t=size(x,1);

% Value close to zero
small=1e-6;

% Allocate output variable
y=NaN*ones(t,1);

% =========================================================================
% TRANSFORMATION:
% Determine case 1-7 by transformation code
switch(tcode)

    case 1 % Level (i.e. no transformation): x(t)
        y=x;
        new_series = strcat(series,{'_lvl'});

    case 2 % First difference: x(t)-x(t-1)
        y(2:t)=x(2:t,1)-x(1:t-1,1);
        new_series = strcat(series,{'_dif_1m'});

    case 3 % Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
        y(3:t)=x(3:t)-2*x(2:t-1)+x(1:t-2);
        new_series = strcat(series,{'_2nd_dif_1m'});

    case 4 % Natural log: ln(x)
        if min(x) < small
            y=NaN;
        else
            y=log(x);
        end
        new_series = strcat(series,{'_log'});

    case 5 % First difference of natural log: ln(x)-ln(x-1)
        if min(x) > small
            x=log(x);
            y(2:t)=x(2:t)-x(1:t-1);
        end
        new_series = strcat(series,{'_dif_log'});

    case 6 % Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
        if min(x) > small
            x=log(x);
            y(3:t)=x(3:t)-2*x(2:t-1)+x(1:t-2);
        end
        new_series = strcat(series,{'_2nd_dif_log'});

    case 7 % First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
        y1(2:t)=(x(2:t)-x(1:t-1))./x(1:t-1);
        y(3:t)=y1(3:t)-y1(2:t-1);
        new_series = strcat(series,{'_dif_pct'});

end

end
