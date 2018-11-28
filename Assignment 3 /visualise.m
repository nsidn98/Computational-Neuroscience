%CNN='CNN_Network';  
%load(CNN);

images = loadMNISTImages('t10k-images.idx3-ubyte');

idx = input('Which image to select:');
selected_image=images(:,:,idx)/255;

%input images

figure(4);
imagesc(images(:,:,idx)/255);
title('Selected Image');
%weights
figure(1);
height=5;
width=4;
for i=1:numFilters
   subplot(height,width,i), imagesc(Wc(:,:,21-i));
   colorbar;
end

%activations
figure(3);
imagesc(selected_image);

figure(2);
height=5;
width=4;
for i=1:numFilters
   activation=cnnConvolve2(filterDim, 1, selected_image, Wc(:,:,i), bc(i));
   subplot(height,width,i), imagesc(activation);
end


