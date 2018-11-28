function y = Softmax(x)
    y = exp(x)/sum(exp(x));
end