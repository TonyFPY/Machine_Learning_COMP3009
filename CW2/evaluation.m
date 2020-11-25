function [recall, precision, f1] = evaluation(predictions, labels)

    tp = sum(predictions == 1 & labels == 1);
    tn = sum(predictions == 0 & labels == 0);
    fp = sum(predictions == 1 & labels == 0);
    fn = sum(predictions == 0 & labels == 1);
    
    if tp == 0 && fn == 0
        recall = -1;
    else
        recall = tp / (tp + fn);
    end
    
    if tp == 0 && fp == 0 
      precision = -1;
    else
      precision = tp / (tp + fp);
    end
    
    if recall == -1 || precision == -1
        f1 = -1;
    else
        if recall + precision == 0
            f1 = 0;
        else
            f1 = 2 * (recall * precision) / (recall + precision);
        end
    end   
end
