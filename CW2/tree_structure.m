function out = tree_structure(t)
    out.op = '';
    out.class = '';
    out.attribute = t.attribute;
    out.threshold = t.threshold;
    
    if isempty(t.kids)
        out.kids = t.kids;
    else
        out.kids = {tree_structure(t.kids{1}), tree_structure(t.kids{2})};
    end
end
