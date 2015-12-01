function [ theta ] = softmaxTrain( x, y, iter, lambda )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n, p] = size(x);
k = max(y);
theta = zeros(p, k);
for k0 = 1:k
    t = 0;
    for i = 1:n
        pYiK = exp(theta(:, k0).'*x(i,:));
        
        t = t - (x(i,:)*(y(i)=k0 - ));
end

