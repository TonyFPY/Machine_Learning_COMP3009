function output = calculateRemainder(p1, n1, p2, n2)
    % This function calculates the remainder, which is the weighted sum of
    % the information of the left and right nodes after the parent node splits.
    
    informationLeft = calculateInformation(p1, n1);
    informationRight = calculateInformation(p2, n2);
    
    totalExamples = p1+n1+p2+n2;
    
    partialInformationLeft = ((p1+n1)/totalExamples)*informationLeft;
    partialInformationRight = ((p2+n2)/totalExamples)*informationRight;
    
    output = partialInformationLeft+partialInformationRight;
    
end
