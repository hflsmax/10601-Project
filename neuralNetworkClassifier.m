function [ label ] = neuralNetworkClassifier( hiddenW, outputW,...
                                              input)
%neuralNetworkClassifier uses hiddenW and outputW as weights and classifies
%input

[inputn, hiddenn] = size(hiddenW);
[~, outputn] = size(outputW);

%compute hiddenOut
hiddenOut = zeros(hiddenn, 1);

for i = 1:hiddenn
    net = sum(hiddenW(:,i).'*input.');
    sigma = 1/(1+exp(-net));
    hiddenOut(i) = sigma;
end

%compute observed output
observed = zeros(outputn, 1);
for i = 1:outputn
    net = sum(outputW(:,i).'*hiddenOut);
    sigma = 1/(1+exp(-net));
    observed(i) = sigma;
end


[~, I] = max(observed);
label = I-1;

end

