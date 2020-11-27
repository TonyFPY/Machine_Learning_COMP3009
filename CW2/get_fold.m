function [train_fold, test_fold] = get_fold(data, fold_idx, num_fold)

    % Return number of rows, i.e. 150 in our iris dataset
    num_sample = size(data, 1); 
    
    % 15 rows per portion for our itis dataset
    portion = floor(num_sample / num_fold);
    
    % Would be 0, 15, 30, ... 135.
    limit = portion * (fold_idx - 1);
    
    % Choose only one fold of data for testing
    test_fold = data(limit+1 : limit+portion, :);
    
    % Choose the other folds of data for training
    train_fold = [data(1:limit, :); data(limit+portion+1 : end, :)];
end
