function outputVec = createCrisisVector(totalDates, finalDate, crisisPeriods)
    % Prepend day to final date and convert to datetime
    finalDate = datetime(['01.' finalDate],'InputFormat','dd.MM.yyyy');
    
    % Create datetime vector from final date to totalDates months back
    dateVec = (finalDate - calmonths(totalDates - 1)):calmonths(1):finalDate;
    
    % Preallocate outputVec with zeros
    outputVec = zeros(1,totalDates);
    
    % Iterate over dateVec
    for i = 1:totalDates
        % Check if dateVec(i) is in crisisPeriods
        if any(dateVec(i) == crisisPeriods)
            outputVec(i) = 1;
        end
    end
end

