function [ label ] = softmaxClassifier( x, theta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[~, k] = size(theta);
maxI = 0;
maxP = 0;
for i = 1:k
    thisP = prob(x, i, theta);
    if (thisP > maxP) 
        maxI = i;
        maxP = thisP;
    end
end
label = maxI - 1;

end

