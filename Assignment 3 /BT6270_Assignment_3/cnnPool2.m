function pooledFeatures = cnnPool2(poolDim, convolvedFeatures)

numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters);
   
  for filterNum = 1:numFilters
     pfeatures = zeros(convolvedDim/poolDim, convolvedDim/poolDim);
     convFeatures= convolvedFeatures(:,:,filterNum);
     x= convolvedDim / poolDim;          
     for i= 0:1:x-1
         p=(i*poolDim)+1; 
         for j= 0:1:x-1             
            q= (j*poolDim)+1;
            r= convFeatures(p:p+(poolDim-1),q:q+(poolDim-1));
            pfeatures(i+1,j+1)= mean(mean(r)); % You can try Max pooling instead of mean pooling
         end
     end
     pooledFeatures(:,:,filterNum)=pfeatures;  
  end
end

