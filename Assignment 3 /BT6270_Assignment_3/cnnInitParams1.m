function [Wc,Wd,bc,bd] = cnnInitParams1(imageDim,filterDim,numFilters,...
                                poolDim,numClasses)
% Initialize parameters for a single set of conv-pooling laters followed by
% fully connected layer
% ---------------------------------------------------------------------------                           
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict%

% Returns:
%  Wc,Wd,bc,bd

%% Initialize parameters randomly based on layer sizes.
assert(filterDim < imageDim,'filterDim must be less that imageDim');
Wc = 1e-1*randn(filterDim,filterDim,numFilters);
outDim = imageDim - filterDim + 1; % dimension of convolved image
assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');  % assume outDim is multiple of poolDim
outDim = outDim/poolDim;
hiddenSize = outDim^2*numFilters;
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1); %  choose weights uniformly from the interval [-r, r]
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
bc = zeros(numFilters, 1);
bd = zeros(numClasses, 1); 

end

