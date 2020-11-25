function decision_tree = decision_tree_learning(features, labels)
    if range(labels) == 0
        decision_tree = init_tree(nan, nan);
        decision_tree.class = labels(1);
    else
        [best_attribute, best_threshold] = choose_attr(features, labels);
        decision_tree = init_tree(best_attribute, best_threshold);
        
        best_attribute 
        best_threshold
        
        subset1_features = features(features(:, best_attribute) <= best_threshold, :);
        subset2_features = features(features(:, best_attribute) > best_threshold, :);
        subset1_labels = labels(features(:, best_attribute) <= best_threshold, :);
        subset2_labels = labels(features(:, best_attribute) > best_threshold, :);

%         subset1_labels
%         subset2_labels
        
        decision_tree.kids = {
            decision_tree_learning(subset1_features, subset1_labels)
            decision_tree_learning(subset2_features, subset2_labels)
        }; 
    end
end


function i = I(a, b)
    if a == 0 || b == 0
        i = 0;
    else
        i = -a * log2(a) - b * log2(b);
    end
end

% Features shape: N * 4
% Labels shape: N * 1
function [best_attribute, best_threshold] = choose_attr(features, labels)
    num_attr = size(features, 2);
    num_samples = size(features, 1);
    num_pos = sum(labels == 1);
    num_neg = num_samples - num_pos;
    best_attribute = 0;
    best_threshold = 0;
    smallest_remainder = inf;
    
    for attr = 1 : num_attr
        feature_col = features(:, attr); % actually transformed to a row
        [sorted, sorted_idx] = sort(feature_col);
%         sorted
        % counter
        % [ t_0 n_0 p_0
        %   t_1 n_1 p_1 ]
        counter = zeros(2, 3);

        for i = 1: num_samples - 1

            counter(1, 1) = counter(1, 1) + 1;
            if labels(sorted_idx(i)) == 1
                counter(1, 3) = counter(1, 3) + 1;
            else
                counter(1, 2) = counter(1, 2) + 1;
            end
            counter(2, 1) = num_samples - counter(1, 1);
            counter(2, 2) = num_neg - counter(1, 2);
            counter(2, 3) = num_pos - counter(1, 3);
            
            % the feature value are same in both side of the current threshhold
            if sorted(i) == sorted(i + 1)
                continue
            end
            
            
            t_0 = counter(1, 1);
            n_0 = counter(1, 2);
            p_0 = counter(1, 3);
                   
            t_1 = counter(2, 1);
            n_1 = counter(2, 2);
            p_1 = counter(2, 3);
           
            remainder = (t_0 / num_samples) * I(p_0 / t_0, n_0 / t_0) + (t_1 / num_samples) * I(p_1 / t_1, n_1 / t_1);
            threshold = (sorted(i) + sorted(i + 1)) / 2 ;
            % info can be nan because of dividing 0, but that's fine
            if remainder < smallest_remainder
                smallest_remainder = remainder;
                best_attribute = attr;
                
                best_threshold = threshold;

            end
        end
    end
end

% function isleaf = is_leaf(tree)
%     isleaf = isempty(tree.kids) && tree.class ~= -1 && is_same()
% end

function tree = init_tree(attr_ind, threshold)
    tree.op = sprintf("attr %d < %f", attr_ind, threshold);
    if isnan(threshold)
        tree.op = 'leaf';
    end
    tree.kids = [];
    tree.class = nan;
    tree.attribute = attr_ind;
    tree.threshold = threshold;
end
