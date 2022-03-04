function output = MAJORITY_VALUE(labels)
    
    numPositive = 0;
    numNegative = 0;
    
    for index = 1 : size(labels, 1)
      
       if labels(index) == 1
           numPositive = numPositive + 1;
       end
       
       if labels(index) == 0
           numNegative = numNegative + 1;
       end
        
    end
    
    if (numNegative > numPositive) && (numNegative/(numNegative + numPositive) >= 0.99)
        output = 0;
    elseif (numNegative < numPositive) && (numPositive/(numNegative + numPositive) >= 0.99)
        output = 1;
    else 
        output = -1;
    end

end