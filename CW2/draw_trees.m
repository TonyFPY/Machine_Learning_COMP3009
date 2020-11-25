function draw_trees(trees, folds, labels)
    mkdir figs
    for f = 1:folds
        for l = 1:labels
            filename = sprintf("tree_fold%dlabel%d", f, l);
            filename = fullfile('figs', filename);
            DrawDecisionTree(tree_structure(trees{f, l}));
            print(gcf, filename, '-djpeg', '-r400')
            close
        end
    end
end
