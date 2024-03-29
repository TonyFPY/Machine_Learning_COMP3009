function [train_fold, test_fold] = get_fold(data, fold_idx, num_fold)

    % Return number of rows, i.e. 150 in our iris dataset
    num_sample = size(data, 1); 
    
    % 4380 rows per portion for our PM2.5 dataset
    portion = floor(num_sample / num_fold);
 
    limit = portion * (fold_idx - 1);
    
    % Choose only one fold of data for testing
    test_fold = data(limit+1 : limit+portion, :);
    
    % Choose the other folds of data for training
    train_fold = [data(1:limit, :); data(limit+portion+1 : end, :)];
end
