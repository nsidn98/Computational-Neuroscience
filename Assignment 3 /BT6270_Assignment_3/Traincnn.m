%%% Written by Anila Gundavarapu and distributed to fulfill the
%%% requirements of the course: 'Principles of Neuroscience'.
%%% CNN network with one 'conv' layer and one 'subsampling' layer
%%% Network was trained on first 10000 MNIST train set and tested on whole test
%%% set. Vary the parametrs such as i) number of filters ii)number of
%%% epoches iii) size of filters to test the power of CNNs.
%%% ==================================================================================

clear all; close all;
%%%% Setting the network Configuration
%%%% ===============================
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)
%%
% Load MNIST Train data
%==========================
%   images = loadMNISTImages('t10k-images.idx3-ubyte');
images = loadMNISTImages('train-images.idx3-ubyte');

images = images(:,:,1:10000);  % Consider only first 10000 images for training 
TrainLabels = loadMNISTLabels( 'train-labels.idx1-ubyte');
labels = 0.*ones(numClasses, size(TrainLabels,1)); %Train_labels is a column vector
    for n = 1: size(TrainLabels, 1)
        labels(TrainLabels(n) + 1, n) = 1;
    end;
labels=labels(:,1:10000);
clear TrainLabels;
%%
% %%%% Visualize  Train data
% % %%%% =========================
% for i1=1:size(images,3)
%    figure(1);imagesc(images(:,:,i1)); 
% end

 %%
% Initialize network weights and params
% ======================================
[Wc,Wd,bc,bd] = cnnInitParams1(imageDim,filterDim,numFilters,poolDim,numClasses);

epochs = 3;
mom = 0;
alpha = 1e-1;
vel_Wc = zeros(size(Wc));
vel_Wd = zeros(size(Wd));
vel_bc = zeros(size(bc));
vel_bd = zeros(size(bd));
%%
% Actual training
%=====================
for e = 1:epochs
  tic;
  startTime = tic();
%Forward Propagation
fprintf('epoch:  %d',e);
numImages = size(images,3);
for i= 1: numImages
    im=images(:,:,i)/255;
  fprintf('Currently training:  Epoch: %d ; Sample: %d / %d\n', e, i,numImages);

  %%% Convolution Layer
convDim = imageDim-filterDim+1; % dimension of convolved output
activations = zeros(convDim,convDim,numFilters);
activations = cnnConvolve2(filterDim, numFilters, im, Wc, bc);

%%%% Pooling Layer
outputDim = (convDim)/poolDim; % dimension of subsampled output
activationsPooled1 = zeros(outputDim,outputDim,numFilters);
activationsPooled1 = cnnPool2(poolDim, activations);

% input to fully connected layer
activationsPooled = reshape(activationsPooled1,1,[]);
clear activationsPooled1;
probs = zeros(numClasses,1);
probs = Softmax(Wd*activationsPooled' + bd);

%Calculate Gradients
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

groundTruth =labels(:,i);
deriv_1 = -(groundTruth - probs);

Wd_grad = deriv_1*activationsPooled;
bd_grad = deriv_1;

deriv_2_pooled_sh = Wd'*deriv_1;

deriv_2_pooled = reshape(deriv_2_pooled_sh,outputDim,outputDim,numFilters);
deriv_2_upsampled = zeros(convDim,convDim,numFilters);

 for filterNum = 1:numFilters

    aux3 = (1/(poolDim^2)).*kron(squeeze(deriv_2_pooled(:,:,filterNum)),ones(poolDim));
    deriv_2_upsampled(:,:,filterNum) = aux3.*activations(:,:,filterNum).*(1-activations(:,:,filterNum));

    f_now = squeeze(deriv_2_upsampled(:,:,filterNum));
    noww = conv2(im,rot90(squeeze(f_now),2),'valid');

    Wc_grad(:,:,filterNum) = squeeze(Wc_grad(:,:,filterNum)) + noww; 
    bc_grad(filterNum) = bc_grad(filterNum) + sum(f_now(:));
   
 end
 clear f_now; clear noww;clear aux3;
 
%update weights along with momentum and learning rate
  vel_Wc  = (mom.* vel_Wc) + (alpha .* Wc_grad);
  Wc = Wc - vel_Wc;
  vel_Wd = (mom.* vel_Wd) + (alpha .* Wd_grad);
  Wd = Wd - vel_Wd;
  vel_bc = (mom.* vel_bc) + (alpha .* bc_grad);
  bc = bc -   vel_bc;
  vel_bd = (mom.* vel_bd) + (alpha .* bd_grad);
  bd = bd - vel_bd;   
end;

 alpha = alpha/2.0;

end
toc;
save CNN_Network;
fprintf('...Done. Training took %.2f seconds\n', toc(startTime));
%%
%Visualize weights
for i=1:20
    figure(1); imagesc(Wc(:,:,i));pause(0.5);
end

%%
 % Test the trained network
 % ===========================
accuracy = testcnn(Wc,Wd,bc,bd,imageDim,filterDim,poolDim,numFilters,numClasses);
fprintf('Accuracy is: %f%%\n',accuracy*100);
