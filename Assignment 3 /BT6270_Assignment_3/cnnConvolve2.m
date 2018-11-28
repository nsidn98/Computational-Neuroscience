function convolvedFeatures = cnnConvolve2(filterDim, numFilters, images, W, b)
 
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;
convolvedFeatures = zeros(convDim, convDim, numFilters); 
  for filterNum = 1:numFilters      
     convolvedImage = zeros(convDim, convDim);
     filter = W(:,:,filterNum);
     filter = rot90(squeeze(filter),2);
     feature =  conv2(images,filter,'valid')+b(filterNum);
     convolvedImage= 1./(1+exp(-feature));         
     convolvedFeatures(:, :, filterNum) = convolvedImage;
  end
end

