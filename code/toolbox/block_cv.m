function [lamopt, lamoptidx] = block_cv(X, Y, alpha, lam_range, min_train_ratio, test_size, h)
[T, N] = size(X);
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

    if alpha == 0

        B_ridge = nan(N,length(lam_range));
        XprimeX = Xn_train' * Xn_train;
        XprimeY = Xn_train'*Yn_train;
        for ii = 1:length(lam_range)
            B_ridge(:, ii) = (XprimeX + lam_range(ii) * eye(N)) \ XprimeY;

            resd = Yn_test - Xn_test*B_ridge(:,ii);
            mse_mat(bl,ii) = mean(resd.^2);
        end
    else

        [B, fitm] = lasso(Xn_train,Yn_train,'Alpha',alpha, 'Lambda',lam_range);
        intercept = fitm.Intercept;

        for lidx = 1:length(lam_range)
            resd = Yn_test - (ones(length(idxTestY),1)*intercept(lidx) ...
                + Xn_test*B(:,lidx));
            mse_mat(bl,lidx) = mean(resd.^2);
        end
    end
end
[~, lamoptidx] = min(mean(mse_mat));
lamopt = lam_range(lamoptidx);
%%
end