function [ theta ] = softmaxTrain( x, y, iter, lambda )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[n, p] = size(x);
k = max(y) + 1;
theta = zeros(p, k);
% newTheta = theta;
for it = 1:iter
    for k0 = 1:k
        t = 0;
        for i = 1:n
            diff = (y(i)==k0) - prob(x(i,:), y(i) + 1, theta);
            t = t - x(i,:)*diff;
        end
        theta(: ,k0) = theta(:, k0) - (lambda * t).';
    end
    theta
%     % check convergence
%     sum(sum(abs(theta - newTheta)))
%     theta = newTheta;
end
end