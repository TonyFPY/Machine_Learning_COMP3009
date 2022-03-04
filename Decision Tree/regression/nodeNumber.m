% Return the number of nodes in a tree
function numNodes = nodeNumber(tree)
      
    if tree.op == ""
        numNodes = 1;
    else
        numNodes = nodeNumber(tree.kids{1}) + nodeNumber(tree.kids{2});
    end
   
end