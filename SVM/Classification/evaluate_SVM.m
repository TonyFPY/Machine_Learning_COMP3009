% Evaluate the SVM that is obtained using testing sets

function precision = evaluate_SVM(predictions, labels)

    length = size(labels, 1);   % Will be 15 due to one fold
    
    matrix = zeros(2);          
    % Obtain a 2x2 matrix
    % TP FP
    % FN TN
    
    for index = 1 : length
       prediction = predictions(index);   
       label = labels(index);
       matrix(2-prediction, 2-label) = matrix(2-prediction, 2-label) + 1;
    end
    
    sum = matrix(2, 2) + matrix(2, 1) + matrix(1, 2) + matrix(1, 1);
    precision = (matrix(1, 1) + matrix(2, 2)) / sum;
    recall = matrix(2, 2) / (matrix(2, 2) + matrix(2, 1));
    f1 = (2 * precision * recall) / (precision + recall); 
    
end