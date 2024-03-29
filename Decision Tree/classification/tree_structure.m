function out = tree_structure(tree)
  
    out.op = tree.op;    
    out.class = tree.prediction;
   
    if (tree.prediction == 1)
        out.class = tree.prediction;
    end
    
    out.attribute = tree.attribute;
    out.threshold = tree.threshold;
    
    if isempty(tree.kids)
        out.kids = tree.kids;
    else
        out.kids = {tree_structure(tree.kids{1}), tree_structure(tree.kids{2})};
    end
end
