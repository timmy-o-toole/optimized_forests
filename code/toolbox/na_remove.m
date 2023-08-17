function data = na_remove(data, opt)

% Cache variables that will be used in function
dat = data.data_trans_outlier_removed;
dates = data.dates;
series = data.series;

% Get index for start dates and end dates
dates = fred_md_date_to_str(dates);
start_d = find(strcmp(dates, opt.start_date)); 
end_d = find(strcmp(dates, opt.end_date));

dat = dat(start_d:end_d, :);
dates = dates(start_d:end_d);

T_insample = ceil((size(dat,1)-opt.m)*opt.min_train_ratio.^2/100)*100;
nan_leading = mean(isnan(dat(1:T_insample,:))) > 0.8;

% Select data range as set above and remove nan values
series = series(~nan_leading);
dat = dat(:, ~nan_leading);

% Perform missing value interpolation method
switch (opt.interpolating_method)
    
    case 'spline'
        % Define options for cubic spline interpolation
        optNaN.method = 2;  % 2: Removes leading and closing zeros if more than 80% of row are NAN, then apply spline
        optNaN.k = 3;       % 3: Uses 7 observations for moving average calculation. Setting for filter(): See remNaN_spline

        % Interpolate missings using spline function
        [dat, pos_nan, index_leading_ending] = remNaNs_spline(dat, optNaN);
        dates(index_leading_ending) = []; 

        % Save function specific results in structure
        data.pos_nan = pos_nan;
        data.index_leading_ending = index_leading_ending;
        
    case 'factor'       
        % Removes leading and closing zeros if more than 80% of row are NAN, 
        % then interpolate missings using static factor model
        [dat, ~, pos_nan, index_leading_ending] = interpolate_missings(dat, opt.nf_static);
        data.mat_missings = pos_nan;
        data.index_leading_ending = index_leading_ending;
        dates(index_leading_ending) = []; 
         
    case 'none'
        % Do nothing
        
end

% Select data range as set above and remove nan values
series = series(~isnan(mean(dat)));
dat = dat(:, ~isnan(mean(dat)));

% Save results in structure
data.data_trans_outlier_and_na_removed = dat;
data.series = series;
data.dates = dates;

end