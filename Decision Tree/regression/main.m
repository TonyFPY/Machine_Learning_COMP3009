% Clear command window and variables
clear;
clc;

% Defining constants
num_sample = -1;
num_labels = 1;
num_folds = 10;

% Loading dataset
disp('Loading dataset...')
data = load("PM2dot5-new.csv");
disp('Dataset loaded.')

% Preparing dataset
randidx = randperm(size(data, 1));
features = data(randidx, 2:10);
labels = data(randidx, 1);

% The 10x1 cell array
decision_trees = cell(num_folds, num_labels);   

for fold = 1 : num_folds
    
    % One fold of data(15) for testing, and nine folds(135) of data for training
    [train_features, test_features] = get_fold(features, fold, num_folds);
    [train_labels, test_labels] = get_fold(labels, fold, num_folds);
    
    % Building decision trees
    for i = 1 : num_labels
        
        fprintf("Building decision tree of fold %d\n",  fold);
        tree = decision_tree_learning(train_features, train_labels);
        
        fprintf("Testing decision tree of fold %d\n", fold);
        rmse = evaluate_tree(tree, test_features, test_labels);

        fprintf("RMSE is %f\n", rmse);
        
        nodes = nodeNumber(tree) - 1;
        depth = depthNumber(tree);
        fprintf("Number of internal nodes is %d, number of depth is %d\n\n", nodes, depth);
        
        decision_trees{fold, i} = tree;
    end
    
end

% numOfnodes = num_nodes(decision_trees);
% sprintf('The number of nodes:%d\n',numOfnodes);
draw_trees(decision_trees, num_folds, num_labels)
