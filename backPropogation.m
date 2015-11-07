function [ newHiddenW, newOutputW ] = backPropogation...
                                    (inputn, hiddenn, outputn, ...
                                    input, output, ...
                                    hiddenW, outputW)
%backPropogation takes the size of the nn, input, output and previous 
%weight of the nn, return the new weight of the nn


assert(length(input) == inputn);
assert(length(output) == outputn);
assert(size(hiddenW) == [inputn, hiddenn]);
assert(size(outputW) == [hiddenn, outputn]);

%compute hiddenOut
hiddenOut = zeros(inputn, hiddenn);
for i = 1:hiddenn
    for j = 1:inputn
        net = sum(hiddenW(i,:)*input);
        sigma = 1/(1+exp(-net));
        hiddenOut(i) = sigma;
    end
end

%compute observed output
observed = zeros(hiddenn, outputn);
for i = 1:outputn
    for j = 1:hiddenn
        net = sum(outputW(i,:)*hiddenOut);
        sigma = 1/(1+exp(-net));
        observed(i) = sigma;
    end
end
        
        
%sigma of output
sigmaOutput = zeros(outputn,1);
for i = 1:outputn
    sigmaOutput(i) = observed(i)*(1-observed(i))*(output(i)-observed(i));
end
%sigma of hidden
sigmaHidden = zeros(hiddenn, 1);
for i = 1:hiddenn
    sigmaHidden(i) = hiddenOut(i)*(1-hiddenOut(i))*sum(hiddenW(i,:)*sigmaOutput);
end
%update hiddenW
newHiddenW = hiddenW;
for i = 1:inputn
    for j = 1:hiddenn
        newHiddenW(i,j) = hiddenW(i,j) + lambda * sigmaHidden(j) * input(i);
    end
end
%update outputW
newOutputW = outputW;
for i = 1:hiddenn
    for j = 1:outputn
        newOutputW(i,j) = outputW(i,j) + lambda * sigmaOutput(j) * hiddenOut(i);
    end
end
    

end

