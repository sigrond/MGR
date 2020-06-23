%test 7 - nauczenie sieci klas ró¿nych substancji i ich stê¿eñ
%bez utraty danych przez skalowanie lub strukturê sieci
%wyko¿ystanie ca³ych klatek
reset(gpuDevice(1));
cz1 = fullfile(...%'minitest');
'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(cz1,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels3;
    save('myImds7d15.mat','imds')
end

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds,...%0.8, 0.1, 0.1 ...
        ...%399,49,49 
         399,50,50 ...
        ,'randomized');
    save('mySets7d15.mat','trainingSet','validationSet','testSet')
end

%conv1 = convolution2dLayer([20 3],38,'Stride',[1 1],'Padding',0);
%conv1.Weights = gpuArray(single(randn([20 3 3 38])*0.00001));
%conv1.Bias = gpuArray(single(randn([1 1 38])*0.00001+1));

conv1=net7d14.Layers(2);
conv1.Weights=conv1.Weights+(rand(11,11,1,96)*60-30);

conv2=net7d14.Layers(5);
conv2.Weights=conv2.Weights+(rand(5,5,96,128)*0.2-0.1);

fc1=net7d14.Layers(8);
fc1.Weights=fc1.Weights+(rand(500,512)*0.2-0.1);

fc2=net7d14.Layers(12);
fc2.Weights=fc2.Weights+(rand(500,500)*0.2-0.1);

fc3=net7d14.Layers(16);
fc3.Weights=fc3.Weights+(rand(60,500)*0.2-0.1);

layers = [
    imageInputLayer([64 64 1],'Normalization', ...
    'none')
    %'zerocenter')
    %warstwa 1
    %conv1
    %convolution2dLayer([11 11],96,'Stride',[4 4],'Padding',0)
    %net7d7.Layers(2)
    conv1
    maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
    %convolution2dLayer([5 5],128,'Stride',[1 1],'Padding',0)
    %net7d7.Layers(5)
    conv2
    reluLayer
    %maxPooling2dLayer([3 3],'Stride',2)
    crossChannelNormalizationLayer(5,'K',2)
%     convolution2dLayer([3 3],192,'Stride',[1 1],'Padding',0)
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
    %fullyConnectedLayer(500)
    %net7d7.Layers(9)
    fc1
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 7
    %fullyConnectedLayer(500)
    %net7d7.Layers(13)
    fc2
    reluLayer
    crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25)
    %warstwa 8
    %fullyConnectedLayer(60)
    %net7c15.Layers(17)
    fc3
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
    'MiniBatchSize', 16, ...
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
    data=frame(1:64,1:64,1);
end
