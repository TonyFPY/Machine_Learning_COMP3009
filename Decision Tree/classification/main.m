% Clear command window and variables
clear;
clc;

% Defining constants
num_sample = -1;
num_labels = 1;
num_folds = 10;

% Loading dataset
disp('Loading dataset...')
data = load("iris_new.csv");
disp('Dataset loaded.')

% Preparing dataset
randidx = randperm(size(data, 1));
features = data(randidx, 1:4);
labels = data(randidx, 5);

% The 10x1 cell array
decision_trees = cell(num_folds, num_labels);   

% The 10x1x3 array
evaluations = zeros(num_folds, num_labels, 3);

for fold = 1 : num_folds
    
    % One fold of data(15) for testing, and nine folds(135) of data for training
    [train_features, test_features] = get_fold(features, fold, num_folds);
    [train_labels, test_labels] = get_fold(labels, fold, num_folds);
    
    % Building decision trees
    for i = 1 : num_labels
        
        fprintf("Building decision tree %d of fold %d\n",  i, fold);
        tree = decision_tree_learning(train_features, train_labels(:, i));
        
        fprintf("Testing decision tree %d of fold %d\n", i, fold);
        [precision, recall, f1] = evaluate_tree(tree, test_features, test_labels(:, i));
        
        evaluations(fold, i, :) = [precision, recall, f1];
        fprintf("Precision is %f, recall is %f, f1-measure is %f\n", precision, recall, f1);
        
        nodes = nodeNumber(tree) - 1;
        depth = depthNumber(tree);
        fprintf("Number of internal nodes is %d, number of depth is %d\n\n", nodes, depth);
        
        decision_trees{fold, i} = tree;
    end
    
end

draw_trees(decision_trees, num_folds, num_labels)