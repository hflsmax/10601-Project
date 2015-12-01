function [ accuracy ] = softmaxTest( data, labels )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here



trainX = data(1:100, :);
trainY = labels(1:100);
testX = data(101:200, :);
testY = labels(101:200, :);

theta = softmaxTrain(im2double(trainX), double(trainY), 10, 0.0001);

result = zeros(100, 1);
for i = 1:100
    result(i) = softmaxClassifier(im2double(testX(i, :)), theta);
end
result
accuracy = sum(result == testY) / 100

end

