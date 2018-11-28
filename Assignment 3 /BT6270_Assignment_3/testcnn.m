%  Test the performance of the trained on MNIST test set.
%=====================================================================
function [accuracy] = testcnn(Wc,Wd,bc,bd,imageDim,filterDim,poolDim,numFilters,numClasses) 
% Load Test set 
testImages = loadMNISTImages('t10k-images.idx3-ubyte');
numtestImages = size(testImages,3);
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
PredLabels = zeros(numtestImages,1);

for i= 1: numtestImages    
    imtest=testImages(:,:,i); 
    % Forward pass
    %Conv layer
    convDim = imageDim-filterDim+1; % dimension of convolved output
    test_activations = zeros(convDim,convDim,numFilters);
    test_activations = cnnConvolve2(filterDim, numFilters, imtest, Wc, bc);
    %pooling layer
    outputDim = (convDim)/poolDim; % dimension of subsampled output
    test_activationsPooled1 = zeros(outputDim,outputDim,numFilters);
    test_activationsPooled1 = cnnPool2(poolDim, test_activations);
    %fully connected layer
    test_activationsPooled = reshape(test_activationsPooled1,1,[]);
    clear test_activationsPooled1;
    test_probs = zeros(numClasses);
    test_probs = Softmax(Wd*test_activationsPooled' + bd);
    %Predict the label
    [~,x] = max(test_probs,[],1);
    PredLabels(i)= x-1;
    fprintf('image: %d   actual: %d   Predicted: %d \n',i,testLabels(i),PredLabels(i));    
end 
accuracy = sum(PredLabels==testLabels)/numtestImages; % Accuracy should be around 90% after 3 epochs
 end
