function output = calculateInformation(p, n)
    % This function calculates the amount of information at a certain node,
    % also known as entropy. The value ranges between 0 and 1. The goal is
    % to have the entropy as low as possible.
    if p == 0 && n == 0
        output = 0;
    else
        a = p/(p+n);
        b = n/(p+n);
        if a == 0
            output = -b*log2(b);
        elseif b == 0
            output = -a*log2(a);
        else
            output = -a*log2(a)-b*log2(b);
        end
    end
end