function [ accuracy ] = neuralNetworkTest(data, labels )
%neuralNetworkTest takes hiddenW and outputW, return the accuracy of the
%model
%   Detailed explanation goes here

% 
% simpleX = zeros(2000, 48);
% for i = 1:2000
%     pixels = data(i,:);
%     for j = 1:(16*3)
%         simpleX(i,j) = mean(pixels(64*(j-1)+1:64*j)) / 255;
%     end
% end

trainX = data(1:100, :);
trainY = labels(1:100);

testX = data(101:200, :);
testY = labels(101:200);

[hiddenW, outputW] = neuralNetworkTraniner(im2double(trainX), ...
                                           trainY);

                                       
result = zeros(100, 1);
for i = 1:100
    result(i) = neuralNetworkClassifier(hiddenW, outputW,...
                                        im2double(testX(i,:)));
end

result
accuracy = (sum(testY == result)/100);

end

