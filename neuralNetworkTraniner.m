function [ hiddenW, outputW ] = neuralNetworkTraniner( x, y )
%neuralNetworkTrainer is a simple nn trainer that use backpropogation,
%using 0 as starting point

% rng(243);
[n, p] = size(x);
hiddenn = 4;
outputn = max(y) + 1;
hiddenW = rand(p, hiddenn);
% sumw = sum(sum(hiddenW));
% for i = 1:hiddenn
%     hiddenW(:, i) = hiddenW(:, i) / sumw;
% end
% repmat(0, p, hiddenn);
outputW = rand(hiddenn, outputn);
% sumw = sum(sum(outputW));
% for i = 1:outputn
%     outputW(:,i) = outputW(:,i) / sumw;
% end
% repmat(0, hiddenn, outputn);
% lambda = 0.0008;
lambda = .1;

for k = 1:50
for i = 1:n
    output = -ones(outputn, 1);
    output(y(i)+1) = 1;
    [hiddenW, outputW] = backPropogation(p, hiddenn, outputn,...
                                         x(i,:), output, ...
                                         hiddenW, outputW, lambda);
                                    
end
hiddenW

end


end

