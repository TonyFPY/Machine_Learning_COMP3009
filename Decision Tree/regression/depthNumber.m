% Return the max depth of a tree
function depth = depthNumber(tree)

    if tree.op == ""
        depth = 0;
    else
        depth = 1 + max(depthNumber(tree.kids{1}), depthNumber(tree.kids{2}));
    end
   
end