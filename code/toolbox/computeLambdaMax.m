% ===================================================
%                 computeLambdaMax() 
% ===================================================
function [lambdaMax,nullMSE]=computeLambdaMax(X0,Y0,weights,alpha)
    
    observationWeights = ~isempty(weights);
    N = size(X0,1);

    % Calculate max lambda that permits non-zero coefficients
    if ~observationWeights
        dotp = abs(X0' * Y0);
        lambdaMax = max(dotp) / (N*alpha);
    else
        wX0 = X0 .* weights';
        dotp = abs(sum(wX0 .* Y0));
        lambdaMax = max(dotp) / alpha;
    end
    
    if ~observationWeights
        nullMSE = mean(Y0.^2);
    else
        % This works because weights are normalized and Y0 is already
        % weight-centered.
        nullMSE = weights * (Y0.^2);
    end
    
end %-computeLambdaMax
