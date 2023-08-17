function [lamopt, lamoptidx] = block_cv_AdaGrid(X, Y, alpha, innit_lambda_range, K, min_train_ratio, test_size, h)
% block_cv_AdaGrid Performs blocked cross-validation with an adaptive grid search for the hyperparameter lambda in ridge regression.
% The function adaptively tunes the lambda hyperparameter by taking the direction of adjustment (increase or decrease)
% for the next lambda based on the position of the smallest mean squared error (MSE) among the three initial lambdas.
% The direction is then adaptively adjusted in subsequent iterations based on whether the new MSE is the minimum so far.
% 
%----------------------------------------------------------------------------------------------------
% Inputs:
%    X, Y - Predictor variables and target variable, respectively.
%    alpha - This parameter is not used in the function.
%    innit_lambda_range - An array of three initial lambda values to test.
%    K - The number of iterations to perform for the grid search.
%    min_train_ratio - The minimum ratio of the dataset to be used for training.
%    test_size - The size of each test block.
%    h - The horizon for which the prediction is made.
%----------------------------------------------------------------------------------------------------
% Outputs:
%    lamopt - Optimal lambda that minimizes the MSE over the blocks.
%    lamoptidx - Index of the optimal lambda in the array of lambda values.
%----------------------------------------------------------------------------------------------------
% The function also contains a nested function ridge_closed_form_multi_lambda which computes ridge regression coefficients 
% for multiple lambda values using the closed form solution.
%
% After the direction of adjustment for lambda is determined, the function continues to adaptively adjust the direction
% in subsequent iterations based on whether the new MSE is the minimum so far. If not, it changes direction.
% Once the direction has been changed, a method similar to a bisection search is employed where the function always takes 
% the mean of the two best lambdas so far for the next lambda.
%----------------------------------------------------------------------------------------------------
% Note: This function performs an adaptive grid search which involves iteratively adjusting the search grid based on previous results. 
% This approach can also be seen as a variant of bisection search with some elements of grid search.
% ===================================================================================================
T = size(X, 1);
train_size = floor(min_train_ratio * T);
numBlocks = floor((T - train_size - 2*h) / test_size);

if (numBlocks < 1)
    error('Test size is too large');
end

if (length(innit_lambda_range) ~= 3)
    error('Lambda Range is NOT 3');
end

mse_mat = nan(numBlocks, K);
innit_mse_mat = nan(numBlocks, length(innit_lambda_range)); %fill in rage
innit_mse_check = nan(numBlocks, 2); % fill in checks
lam_range = nan(1, K);

directionChanged = false; % Track if direction has changed

for i = 1:K

    % Create Errors for all Block for Lambda_i
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

        if i == 1 % Initialise at first iteration
            
            % Add Innits for Sign Changes, allocate min max
            innit_lambda_range(4) = innit_lambda_range(1) * 1.05; %check.min
            innit_lambda_range(5) = innit_lambda_range(3) *0.095; %check.max

            % USE PREDICTION METHOD
            B = ridge_closed_form_multi_lambda(Xn_train, Yn_train, innit_lambda_range); 
            resd = Yn_test - Xn_test*B;
            innit_mse_mat(bl,:) = mean(resd(:,1:3).^2,1);
            innit_mse_check(bl,:) =  mean(resd(:,4:5).^2,1);

        else % Regular MSE calculation
            % USE PREDICTION METHOD
            B = ridge_closed_form_multi_lambda(Xn_train, Yn_train, lam_range(i));
            resd = Yn_test - Xn_test*B;
            mse_mat(bl,i) = mean(resd.^2);
        end
    end

    % Initalise the Lambda and the direction (can be deleted)
    if i == 1 % Select the initial lambda and direction after the first iteration
        avg_mse = mean(innit_mse_mat, 1); % Average MSE for each initial lambda
        [~, sortedIndices] = sort(avg_mse); % Sort MSEs to find smallest
        avg_mse_check = mean(innit_mse_check, 1);

        % Determine the direction of lambda adjustment based on where the smallest MSE falls
        if sortedIndices(3) == 1 % If largest lambda gave smallest MSE
            direction = 0; % We should increase lambda in the future

            % Check: Is maximum to strong?
            if avg_mse(3) > avg_mse_check(2) 
                direction = 1; % decrease
            end 

        elseif sortedIndices(1) == 1 % If smallest lambda gave smallest MSE
            direction = 1; % We should decrease lambda in the future

            % Check: If minimum to weak?
            if avg_mse(1) > avg_mse_check(1) 
                direction = 0; % increase
            end

        else % If the middle lambda gave smallest MSE
            if sortedIndices(2) == 1 && sortedIndices(3) == 2 % If the smallest MSE came from the second lambda
                direction = 0; % We should increase lambda in the future
            else % If the second smallest MSE came from smallest lambda
                direction = 1; % We should decrease lambda in the future
            end
        end

        lam_range(1) = innit_lambda_range(sortedIndices(1)); % Set the initial lambda to the one that gave smallest MSE
        mse_mat(:,1) = innit_mse_mat(:,sortedIndices(1)); % Record the corresponding MSE
    end



 % Converge direction
    if i < K

        if ~directionChanged
            % Generate NEW Strong Lambda if on the outer Edge
            if mean(mse_mat(:,i)) == min(mean(mse_mat(:,1:i))) && direction == 0
                lam_range(i+1) = lam_range(i) *1.5;
                direction = 0; % Increase
            elseif mean(mse_mat(:,i)) == min(mean(mse_mat(:,1:i))) && direction == 1
                lam_range(i+1) = lam_range(i) * 0.5;
                direction = 1; % Decrease

            % If new MSE is not the optimum, do:
            else
                % We increased lambda, but failed, hence decrease next
                if direction == 0
                    directionChanged = true;
                    direction = 1; % Decrease
                else
                    % We decreased the lambda, but failed, hence we increase
                    % next
                    directionChanged = true;
                    direction = 0; % Increase
                end
            end
        end

        % If direction has been changed once, always take mean of best two lambdas
        if directionChanged
            [~,idx] = sort(mean(mse_mat(:,1:i)));
            topTwo = lam_range(idx(1:2));
            lam_range(i+1) = mean(topTwo);
        end
    end 
        
    end % End K Loop 

    [~, lamoptidx] = min(mean(mse_mat));
    lamopt = lam_range(lamoptidx);

end




    function B = ridge_closed_form_multi_lambda(X, y, lambda_vec)
        % Compute the number of predictors
        p = size(X, 2);

        % Initialize the coefficient matrix
        B = zeros(p, length(lambda_vec));

        % Loop over each lambda value
        for i = 1:length(lambda_vec)
            % Compute the coefficients for this lambda value
            B(:, i) = inv(X' * X + lambda_vec(i) * eye(p)) * X' * y;
        end
    end