function [ hiddenW, outputW ] = neuralNetworkTraniner( x, y )
%neuralNetworkTrainer is a simple nn trainer that use backpropogation,
%using 0 as starting point

[n, p] = size(x);
hiddenn = 10;
outputn = 10;
hiddenW = repmat(0, p, hiddenn);
outputW = repmat(0, hiddenn, outputn);
lambda = 0.01;


for i = 1:n
    output = zeros(outputn, 1);
    output(y(i)+1) = 1;
    [hiddenW, outputW] = backPropogation(p, hiddenn, outputn,...
                                         x(i,:), output, ...
                                         hiddenW, outputW, lambda);
    %outputW(1:5, :)
end


end

