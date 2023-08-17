function [lamopt, lamoptidx] = block_cv_adaptive(X, Y, alpha, lam_range, min_train_ratio, test_size, h, opt)
T = size(X, 1);
train_size = floor(min_train_ratio * T); % Get traning size
numBlocks = floor((T - train_size - 2*h) / test_size); % Compute number of blocks

if (numBlocks < 1)
    error('Test size is too large');
end

mse_mat = nan(numBlocks, length(lam_range));
for bl = 1:numBlocks

    idxTrainX = (bl-1)*test_size+1:train_size+(bl-1)*test_size;
    idxTrainY = (bl-1)*test_size+h+1:train_size+(bl-1)*test_size+h;

    if bl == numBlocks
        idxTestX = train_size+(bl-1)*test_size+h+1:T-h;
        idxTestY = train_size+(bl-1)*test_size+2*h+1:T;
    else
        idxTestX = train_size+(bl-1)*test_size+h+1:train_size+bl*test_size+h;
        idxTestY = train_size+(bl-1)*test_size+2*h+1:train_size+bl*test_size+2*h;
    end
    
    X_mean = mean(X(idxTrainX,:));
    X_std = std(X(idxTrainX,:));
    Xn_train = (X(idxTrainX,:) - X_mean) ./ X_std;

    Y_mean = mean(Y(idxTrainY,1));
    Y_std = std(Y(idxTrainY,1));
    Yn_train = (Y(idxTrainY,1) - Y_mean) ./ Y_std;

    Xn_test = (X(idxTestX,:) - X_mean) ./ X_std;
    Yn_test = (Y(idxTestY,1) - Y_mean) ./ Y_std;

    % Add two step lasso for adaptive option
    
    % Get lambda opt through CV
    B_adaptive = lasso_adaptive(Xn_train, Yn_train, alpha, lam_range, min_train_ratio, opt, h);

    for lidx = 1:length(lam_range)
        resd = Yn_test - Xn_test*B_adaptive(:,lidx);
        mse_mat(bl,lidx) = mean(resd.^2);
    end

end
[~, lamoptidx] = min(mean(mse_mat));
lamopt = lam_range(lamoptidx);

end