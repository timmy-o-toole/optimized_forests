function data = data_transform_all(data)
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
%           data     = transformed data
%
% -------------------------------------------------------------------------
% SUBFUNCTION:
%           transxf:    transforms a single series as specified by a
%                       given transformation code
%
% =========================================================================

% Initialize output variable
yt        = [];
st        = [];
dat = data.data_raw;
series = data.series;

% Define number of different transformations
transform_iter = [1:1:12];

% Perform transformation using subfunction transxf (see below for details)
for i = 1:max(transform_iter)
    for j = 1:2
        [dum, new_ser] = transxf(dat(:,:), series, i, j);
        yt = [yt, dum];
        st = [st, new_ser];
    end
end

% Delete series with only NAN values
idx_nan = all(isnan(yt));
yt = yt(:,~idx_nan); %cols that are all nan
st = st(:,~idx_nan);

% Save results in structure
data.data_trans = yt;
data.series = st;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION

function [y, new_series]=transxf(x, series, i, j)
% =========================================================================
% DESCRIPTION:
% This function transforms a single series (in a column vector)as specified
% by a given transfromation code.
%
% -------------------------------------------------------------------------
% INPUT:
%           x        = series (in a column vector) to be transformed
%           series   = series names
%
% OUTPUT:
%           y          = transformed series (as a column vector)
%           new_series = new vector of series names
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

    case 8 % Year over year percent change
        y(13:t,:) = (x(13:t,:) -x(1:t-12,:))./abs(x(1:t-12,:))*100 ;
        new_series = strcat(series,{'_pct_1y'});

    case 9 % Year over year change
        y(13:t,:) = (x(13:t,:) -x(1:t-12,:)) ;
        new_series = strcat(series,{'_dif_1y'});

    case 10 % quarter over quarter percent change
        y(4:t,:) = (x(4:t,:) -x(1:t-3,:))./abs(x(1:t-3,:))*100 ;
        new_series = strcat(series,{'_pct_1q'});

    case 11 % quarter over quarter change
        y(4:t,:) = (x(4:t,:) -x(1:t-3,:)) ;
        new_series = strcat(series,{'_dif_1q'});

    case 12 % Box-Cox Transformation
        % Note: Could also be extended to Yeo-Johnson Transformation for
        % negative values, but seems to be not so well known.
        N = size(x,2);
        for ii = 1:N
            if min(x(:,ii)) > small
                y(:,ii) = boxcox(x(:,ii));
            else
                y(:,ii) = NaN;
            end
            new_series(1,ii) = strcat(series(ii),{'_box'});
        end

end

switch(j)

    case 1 % Do nothing

    case 2 % Detrended using a moving average over the last 12 observations
        movingmean = movmean(y,[11 0]); % 11 steps in the past plus the last observation
        func = @(y)max(find((1:size(y,1))'==cumsum(isnan(y(:,1)))));
        leadZero = splitapply(func, y, 1:size(y,2));
        leadZero = min(leadZero);
        if( isempty(leadZero) )
            leadZero = 0;
        end
        y(13+leadZero:t,:) = y(13+leadZero:t,:) - movingmean(12+leadZero:t-1,:);
        new_series = strcat(new_series,{'_detrended'});

end

end
