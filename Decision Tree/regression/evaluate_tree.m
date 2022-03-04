% Evaluate the tree that is obtained using testing sets
function rmse = evaluate_tree(tree, features, labels)

    length = size(labels, 1);   % Will be 4380 due to one fold
        
    % Obtain a 2x2 matrix
    % TP FP
    % FN TN
    rmse = ones(length,1);
    for index = 1 : length
       sample = features(index, :);
       prediction = predict(tree, sample);    
       label = labels(index);
       
       rmse(index) = sqrt(mean((prediction-label).^2));

    end
    
    rmse = mean(rmse);
  
    
end

% Predict a sample's result
function class = predict(tree, sample)

    node = tree;
    while node.op ~= ""
        threshold = node.threshold;
        value = sample(node.attribute);
        if value < threshold
           node = node.kids{1};
        else
           node = node.kids{2};
        end
        
    end
            
    class = node.prediction;
    
end