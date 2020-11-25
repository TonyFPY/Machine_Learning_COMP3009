function [precision, recall, f1] = evaluate_tree(tree, features, labels)
    length = size(labels, 1);
    mat = zeros(2);
    for index = 1 : length
       sample = features(index, :);
       prediction = predict(tree, sample);
       label = labels(index);
       mat(prediction + 1, label + 1) = mat(prediction + 1, label + 1) + 1;
    end
    precision = mat(2, 2) / (mat(2, 2) + mat(2, 1));
    recall = mat(2, 2) / (mat(2, 2) + mat(1, 2));
    f1 = 2 * precision * recall / (precision + recall); 
    
end


function class = predict(tree, sample)
    node = tree;
    cur = node;
    while cur.op ~= "leaf"
        threshold = cur.threshold;
        value = sample(cur.attribute);
        if value < threshold
           cur = cur.kids{1};
        else
           cur = cur.kids{2};
        end
        
    end
            
    class = cur.class;
    
end
