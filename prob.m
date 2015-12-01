function [ p ] = prob( x, y, theta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


[~, k] = size(theta);

denom = 0;
for i = 1:k
    denom = denom + exp(x * theta(:, i));
end

p0 = exp(x * theta(:, y)) / denom;
if (isnan(p0)) 
    p = 1/k;
else p = p0;

end

