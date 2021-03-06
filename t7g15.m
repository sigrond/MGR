%test 7 - nauczenie sieci klas r�nych substancji i ich st�e�
%bez utraty danych przez skalowanie lub struktur� sieci
%wyko�ystanie ca�ych klatek
reset(gpuDevice(1));
cz1 = fullfile(...%'minitest');
'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(files,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels3;
    save('myImds\myImds7g16.mat','imds')
end

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds,...%0.8, 0.1, 0.1 ... 
        ...%399,49,49 ...
         100,10,10 ...
        ,'randomized');
    save('mySets\mySets7g16.mat','trainingSet','validationSet','testSet')
end

%conv1 = convolution2dLayer([20 3],38,'Stride',[1 1],'Padding',0);
%conv1.Weights = gpuArray(single(randn([20 3 3 38])*0.00001));
%conv1.Bias = gpuArray(single(randn([1 1 38])*0.00001+1));

% conv1=net7d16.Layers(2);
% conv1.Weights=conv1.Weights+(rand(11,11,1,96)*20-10);
% 
% conv2=net7d16.Layers(5);
% conv2.Weights=conv2.Weights+(rand(5,5,96,128)*0.01-0.005);
% 
% fc1=net7e1.Layers(8);
% fc1o=net7d16.Layers(8);
% %fc1.Weights=fc1.Weights+(rand(500,512)*0.01-0.005);
% fc1.Weights(1:500,1:512)=fc1o.Weights;
% 
% fc2=net7d16.Layers(12);
% fc2.Weights=fc2.Weights+(rand(500,500)*0.01-0.005);
% 
% fc3=net7d16.Layers(16);
% fc3.Weights=fc3.Weights+(rand(60,500)*0.01-0.005);

layers = [
    imageInputLayer([480 640 1],'Normalization', ...
    'none')
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    reluLayer
    crossChannelNormalizationLayer(5,'K',2)
    maxPooling2dLayer([3 3],'Stride',2)
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    reluLayer
    crossChannelNormalizationLayer(5,'K',2)
    maxPooling2dLayer([3 3],'Stride',2)
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    batchNormalizationLayer
    reluLayer
    %crossChannelNormalizationLayer(5,'K',2)
    maxPooling2dLayer([3 3],'Stride',2)
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    %crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
%     reluLayer
%     maxPooling2dLayer([3 3],'Stride',2)
%     crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
%     reluLayer
%     maxPooling2dLayer([3 3],'Stride',2)
%     crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
%     reluLayer
%     maxPooling2dLayer([3 3],'Stride',2)
%     crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
%     reluLayer
%     crossChannelNormalizationLayer(5,'K',2)
    %warstwa 6
    %fullyConnectedLayer(500)
    fullyConnectedLayer(512)
    %net7d7.Layers(9)
    %fc1
    batchNormalizationLayer
    reluLayer
    %crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 7
    fullyConnectedLayer(256)
    %net7d7.Layers(13)
    %fc2
    reluLayer
    %crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 8
    fullyConnectedLayer(53)
    %net7c15.Layers(17)
    %fc3
    softmaxLayer
    classificationLayer
 ]

%imageSize = [20 640 3];
%augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet);
%augmentedValidationSet = augmentedImageDatastore(imageSize, validationSet);
%augmentedTestSet = augmentedImageDatastore(imageSize, testSet);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationSet, ...
    'ValidationFrequency',1024, ...
    'ValidationPatience',40, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10,...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MiniBatchSize', 8, ...
    'CheckpointPath', 't7c');

%load 'myAlexNet.mat';
if ~exist('net', 'var')
    net = trainNetwork(trainingSet,layers,options);
else
    net = trainNetwork(trainingSet,net.Layers,options);
end

[YPred,scores] = classify(net,testSet,'MiniBatchSize',1);
[S,I] = maxk(scores',5);
YValidation = testSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

top5 = sum(sum(tbl.Label(I)' == YValidation))/numel(YValidation)

function data = customreader(filename)
    load(filename,'frame');
    %x=randi(10);
    %y=randi(10);
    f64=frame(1:480,1:640,1);
    mf64=double(max(max(max(f64))));
    data=double(f64)./mf64;
end
