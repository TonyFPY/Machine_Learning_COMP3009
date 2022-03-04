function decision_tree = decision_tree_learning(features, labels)

    if MAJORITY_VALUE(labels) == 1
    % If the size of labels are less than a pre-defined value
    % Then it reaches a leaf node
        decision_tree = initializeTree(nan, nan);
        decision_tree.prediction = mean(labels);
    else

        sprintf('%.f\n',var(labels));
        [best_attribute, best_threshold] = choose_attribute(features, labels);
        decision_tree = initializeTree(best_attribute, best_threshold);
               
        subset1_features = features(features(:, best_attribute) <= best_threshold, :);
        subset2_features = features(features(:, best_attribute) > best_threshold, :);
        subset1_labels = labels(features(:, best_attribute) <= best_threshold, :);
        subset2_labels = labels(features(:, best_attribute) > best_threshold, :);

        decision_tree.kids = {
            decision_tree_learning(subset1_features, subset1_labels)
            decision_tree_learning(subset2_features, subset2_labels)
        }; 
    end
end

% Features shape: N * 4
% Labels shape: N * 1
function [best_attribute, best_threshold] = choose_attribute(features, labels)

    num_attribute = size(features, 2); % Will be 4 for our iris dataset
    num_samples = size(features, 1); % Number of samples
    num_pos = num_samples/2; % Number of positive samples
    num_neg = num_samples - num_pos; % Number of negative samples
    information = calculateInformation(num_pos, num_neg);
    best_attribute = 0;
    best_threshold = 0;
    highest_gain = -inf;
    
    for attribute = 1 : num_attribute
        
        % Obtain a column of features
        feature_column = features(:, attribute);    
        
        % Obtain the sorted column and its corresponding index
        [sorted, sorted_idx] = sort(feature_column);   

        % The 2x3 Counter for calcaulating number of postive and negative examples
        % counter(1,3) for positive examples counted
        % counter(1,2) for negative examples counted
        counter = zeros(2, 3);

        for i = 1: num_samples - 1

            counter(1, 1) = counter(1, 1) + 1;
            
            if sorted_idx(i) <= num_pos              
                counter(1, 3) = counter(1, 3) + 1;
            else
                counter(1, 2) = counter(1, 2) + 1;
            end
            
            % The remaining number of samples
            counter(2, 1) = num_samples - counter(1, 1);
            
            % The remaining number of negative samples
            counter(2, 2) = num_neg - counter(1, 2);
            
            % The remaining number of positive samples
            counter(2, 3) = num_pos - counter(1, 3);
            
            % The feature values are the same in both sides of the current threshold
            if sorted(i) == sorted(i + 1)
                continue
            end
            
            n_0 = counter(1, 2);
            p_0 = counter(1, 3);
                   
            n_1 = counter(2, 2);
            p_1 = counter(2, 3);
           
            remainder = calculateRemainder(p_0, n_0, p_1, n_1);
            threshold = (sorted(i) + sorted(i + 1)) / 2 ;

            
            
            gain = information - remainder;

            if gain > highest_gain
                highest_gain = gain;
                best_attribute = attribute;  
                best_threshold = threshold;
            end
            
            subset1_features = features(features(:, best_attribute) <= best_threshold, :);
            subset2_features = features(features(:, best_attribute) > best_threshold, :);
            
            if size(subset1_features)<20 | size(subset2_features) <20
                decision_tree = initializeTree(nan, nan);
                decision_tree.prediction = mean(labels);
            end
        end
    end
end


% Initialize a tree
function tree = initializeTree(attr_ind, threshold)

    if attr_ind == 1
       tree.op = sprintf("DEWP(%d) <= %.3f", attr_ind, threshold);
    elseif attr_ind == 2
       tree.op = sprintf("TEMP(%d)<= %.3f", attr_ind, threshold); 
    elseif attr_ind == 3
       tree.op = sprintf("PRES(%d)<= %.3f", attr_ind, threshold);  
    elseif attr_ind == 4
       tree.op = sprintf("cbwb(%d): %d", attr_ind, threshold);
    elseif attr_ind == 5
       tree.op = sprintf("lws(%d): %d", attr_ind, threshold);
    elseif attr_ind == 6
       tree.op = sprintf("ls(%d): %d", attr_ind, threshold);
    elseif attr_ind == 7
       tree.op = sprintf("lr(%d): %d", attr_ind, threshold);
    elseif attr_ind == 8
       tree.op = sprintf("season(%d): %d", attr_ind, threshold);
    elseif attr_ind == 9
       tree.op = sprintf("time(%d): %d", attr_ind, threshold);
    end
    
    if isnan(threshold)
        tree.op = '';
    end
    tree.kids = [];
    tree.prediction = nan;
    tree.attribute = attr_ind;
    tree.threshold = threshold;
end