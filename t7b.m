%test 7 - nauczenie sieci klas ró¿nych substancji i ich stê¿eñ
%bez utraty danych przez skalowanie lub strukturê sieci
%wyko¿ystanie ca³ych klatek
reset(gpuDevice(1));
%cz1 = fullfile('@26.9deg_shutter in microsec');
%imds = imageDatastore(cz1,'LabelSource','foldernames','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));

%match_labels2;
save('myImds.mat','imds')

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds, 0.8, 0.1, 0.1, 'randomized');
    save('mySets.mat','trainingSet','validationSet','testSet')
end

%conv1 = convolution2dLayer([20 3],38,'Stride',[1 1],'Padding',0);
%conv1.Weights = gpuArray(single(randn([20 3 3 38])*0.00001));
%conv1.Bias = gpuArray(single(randn([1 1 38])*0.00001+1));

layers = [
    imageInputLayer([480 640 2],'Normalization', ...
    'none')
    %'zerocenter')
    %warstwa 1
    %conv1
    convolution2dLayer([5 5],32,'Stride',[1 1],'Padding',0)
    reluLayer
    maxPooling2dLayer([5 5],'Stride',3)
    crossChannelNormalizationLayer(5,'K',2)
    convolution2dLayer([5 5],32,'Stride',[1 1],'Padding',0)
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    convolution2dLayer([5 5],32,'Stride',[1 1],'Padding',0)
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    convolution2dLayer([5 5],32,'Stride',[1 1],'Padding',0)
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
    reluLayer
    crossChannelNormalizationLayer(5,'K',2)
    %warstwa 6
    fullyConnectedLayer(1024)
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer
    %warstwa 7
    fullyConnectedLayer(1024)
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer
    %warstwa 8
    fullyConnectedLayer(64)
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
    'ValidationFrequency',4086, ...
    'ValidationPatience',7, ...
    'LearnRateSchedule','piecewise', ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MiniBatchSize', 16, ...
    'CheckpointPath', 't7b');

%load 'myAlexNet.mat';
%if ~exist('net', 'var')
    net = trainNetwork(trainingSet,layers,options);
%end

[YPred,scores] = classify(net,testSet,'MiniBatchSize',16);
[S,I] = maxk(scores',5);
YValidation = testSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

top5 = sum(sum(tbl.Label(I)' == YValidation))/numel(YValidation)

function data = customreader(filename)
    load(filename,'frame');
    data=squeeze(frame(1,:,:,2:3));
end
