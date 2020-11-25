clear all;
clc;
close all;

% Defining constants
num_sample = -1;
num_labels = 1;
num_folds = 10;
debug_single_fold = false;

% Loading dataset
disp('Loading dataset...')
if ~ (exist('label_data', 'var'))
    data = load("data_new.csv");
    disp('Dataset loaded.')
end

% Preparing dataset
randidx = randperm(size(data, 1));
features = data(randidx, 1:4);
labels = data(randidx, 5);

% if num_sample ~= -1
%     features = features(1:num_sample, :);
%     labels = labels(1:num_sample, :);
% end

decision_trees = cell(num_folds, num_labels);
evaluations = zeros(num_folds, num_labels, 3);

for fold = 1 : num_folds
    [train_features, test_features] = get_fold(features, fold, num_folds);
    [train_labels, test_labels] = get_fold(labels, fold, num_folds);
    
    % Building decision trees

    for i = 1 : num_labels
        fprintf("Building decision tree %d of fold %d\n",  i, fold);
        tree = decision_tree_learning(train_features, train_labels(:, i));
        fprintf("Testing decision tree %d of fold %d\n", i, fold);
        [p, r, f1] = evaluate_tree(tree, test_features, test_labels(:, i));
        evaluations(fold, i, :) = [p, r, f1];
        decision_trees{fold, i} = tree;
    end
    if debug_single_fold
        break
    end
end

draw_trees(decision_trees, num_folds, num_labels)
