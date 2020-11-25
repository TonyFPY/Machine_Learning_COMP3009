function [train_fold, test_fold] = get_fold(data, fold_idx, num_fold)
    num_sample = size(data, 1);
    portion = floor(num_sample / num_fold);
    limit = portion * (fold_idx - 1);
    test_fold = data(limit+1:limit+portion, :);
    train_fold = [data(1:limit, :); data(limit+portion+1:end, :)];
end
