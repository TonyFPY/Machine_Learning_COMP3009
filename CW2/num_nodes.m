function n = num_nodes(tree)
    if tree.op == "leaf"
        n = 1;
    else
        n = num_nodes(tree.kids{1}) + num_nodes(tree.kids{2});
    end
end
