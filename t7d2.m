%test 7 - nauczenie sieci klas ró¿nych substancji i ich stê¿eñ
%bez utraty danych przez skalowanie lub strukturê sieci
%wyko¿ystanie ca³ych klatek
reset(gpuDevice(1));
cz1 = fullfile('minitest');
%'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(cz1,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels3;
    save('myImds7d1.mat','imds')
end

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds,0.8, 0.1, 0.1 ...
        ...%399,49,49 ...
        ,'randomized');
    save('mySets7d1.mat','trainingSet','validationSet','testSet')
end

%conv1 = convolution2dLayer([20 3],38,'Stride',[1 1],'Padding',0);
%conv1.Weights = gpuArray(single(randn([20 3 3 38])*0.00001));
%conv1.Bias = gpuArray(single(randn([1 1 38])*0.00001+1));

layers = [
    imageInputLayer([480 640 1],'Normalization', ...
    'none')
    %'zerocenter')
    %warstwa 1
    %conv1
    %convolution2dLayer([11 11],96,'Stride',[4 4],'Padding',0)
    layers_1(2)
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    %convolution2dLayer([5 5],128,'Stride',[1 1],'Padding',0)
    layers_1(5)
    reluLayer
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
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
%     maxPooling2dLayer([3 3],'Stride',2)
%     crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],32,'Stride',[1 1],'Padding',0)
%     reluLayer
%     crossChannelNormalizationLayer(5,'K',2)
    %warstwa 6
    fullyConnectedLayer(500)
    %net7c15.Layers(9)
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 7
    fullyConnectedLayer(500)
    %net7c15.Layers(13)
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 8
    fullyConnectedLayer(4)
    %net7c15.Layers(17)
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
    'ValidationPatience',10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',20,...
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
    data=frame(1:480,1:640,1);
end
