% Draw a series of figures of decision trees
function draw_trees(trees, folds, labels)

    folder = './figures/';
    if ~exist(folder,'dir')
       mkdir(folder);
    end
     
    for fold = 1 : folds
        for label = 1 :labels
  
            filename = sprintf("tree_fold%d-label%d", fold, label);
            filename = fullfile('figures', filename);      
            DrawDecisionTree(tree_structure(trees{fold, label}));         
            print(gcf, filename, '-djpeg', '-r400')
            
            close
        end
    end
end