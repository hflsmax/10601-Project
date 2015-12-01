function [ hiddenW, outputW ] = neuralNetworkTraniner( x, y )
%neuralNetworkTrainer is a simple nn trainer that use backpropogation,
%using 0 as starting point

% rng(1);
[n, p] = size(x);
hiddenn = 50;
outputn = 10;
hiddenW = rand(p, hiddenn);
for i = 1:hiddenn
    hiddenW(:, i) = hiddenW(:, i) / sum(hiddenW(:, i));
end
% repmat(0, p, hiddenn);
outputW = rand(hiddenn, outputn);
for i = 1:outputn
    outputW(:,i) = outputW(:,i) / sum(outputW(:,i));
end
% repmat(0, hiddenn, outputn);
% lambda = 0.0008;
lambda = .00001;


for i = 1:n
    output = zeros(outputn, 1);
    output(y(i)+1) = 1;
    [hiddenW, outputW] = backPropogation(p, hiddenn, outputn,...
                                         x(i,:), output, ...
                                         hiddenW, outputW, lambda);
end


end

