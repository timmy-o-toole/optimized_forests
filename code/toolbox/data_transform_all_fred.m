function data = data_transform_all_fred(data)
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

% Initialize output variable
yt = [];
st = [];
dat = data.data_raw;
series = data.series;

% Define number of different transformations
transform_iter = [1:1:7];

% Perform transformation using subfunction transxf (see below for details)
for i = 1:max(transform_iter)
    [dum, new_ser] = transxf(dat(:,:), i, series);
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

function [y, new_series]=transxf(x, i, series)
% =========================================================================
% DESCRIPTION:
% This function transforms a single series (in a column vector)as specified
% by a given transfromation code.
%
% -------------------------------------------------------------------------
% INPUT:
%           x       = series (in a column vector) to be transformed
%           i       = iterator over each transformation
%           series  = series names
%
% OUTPUT:
%           y          = transformed series (as a column vector)
%           new_series = new series names
%
% =========================================================================
% SETUP:
% Number of observations (including missing values)
t=size(x,1);
n=size(x,2);

% Value close to zero
small=1e-6;

% Allocate output variable
y=NaN*ones(t,n);

% =========================================================================
% TRANSFORMATION:
% Determine case 1-7 by transformation code
switch(i)

    case 1 % Level (i.e. no transformation): x(t)
        y=x;
        new_series = strcat(series,{'_lvl'});

    case 2 % First difference: x(t)-x(t-1)
        y(2:t,:)=x(2:t,:)-x(1:t-1,:);
        new_series = strcat(series,{'_dif_1m'});

    case 3 % Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
        y(3:t,:)=x(3:t,:)-2*x(2:t-1,:)+x(1:t-2,:);
        new_series = strcat(series,{'_2nd_dif_1m'});

    case 4 % Natural log: ln(x)
        N = size(x,2);
        %new_series = NaN*ones(1,N);
        for ii = 1:N
            if min(x(:,ii)) < small
                y(:,ii) = NaN;
            else
                y(:,ii) = log(x(:,ii));
            end
            new_series(1,ii) = strcat(series(ii),{'_log'});
        end

    case 5 % First difference of natural log: ln(x)-ln(x-1)
        N = size(x,2);
        for ii = 1:N
            if min(x(:,ii)) > small
                x(:,ii) = log(x(:,ii));
                y(2:t,ii) = x(2:t,ii)-x(1:t-1,ii);
            else
                y(:,ii) = NaN;
            end
            new_series(1,ii) = strcat(series(ii),{'_dif_log'});
        end

    case 6 % Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
        N = size(x,2);
        for ii = 1:N
            if min(x(:,ii)) > small
                x(:,ii) = log(x(:,ii));
                y(3:t,ii) = x(3:t,ii)-2*x(2:t-1,ii)+x(1:t-2,ii);
            else
                y(:,ii) = NaN;
            end
            new_series(1,ii) = strcat(series(ii),{'_2nd_dif_log'});
        end

    case 7 % First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
        y1(2:t,:)=(x(2:t,:)-x(1:t-1,:))./x(1:t-1,:);
        y(3:t,:)=y1(3:t,:)-y1(2:t-1,:);
        new_series = strcat(series,{'_dif_pct'});

end

end
