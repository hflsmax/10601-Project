function [ accuracy ] = neuralNetworkTest(data, labels )
%neuralNetworkTest takes hiddenW and outputW, return the accuracy of the
%model
%   Detailed explanation goes here


simpleX = zeros(2000, 48);
for i = 1:2000
    pixels = data(i,:);
    for j = 1:(16*3)
        trainX(i,j) = mean(pixels(64*(j-1)+1:64*j));
    end
end

trainX = simpleX(1:1000);
trainY = labels(1:1000);

testX = simpleX(1001:2000, :);
testY = labels(1001:2000);

[hiddenW, outputW] = neuralNetworkTraniner(im2double(trainX), ...
                                           trainY);

                                       
result = zeros(100, 1);
for i = 1:100
    result(i) = neuralNetworkClassifier(hiddenW, outputW,...
                                        im2double(testX(i,:)));
end

result
accuracy = sum(testY == result);

end

