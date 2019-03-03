%test 1 - nauczenie sieci na 3 pierwszych filmach do klasyfikowania do tych
%próbek?

cz1 = fullfile('cz1');
imds = imageDatastore(cz1,'LabelSource','foldernames','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));

tbl = countEachLabel(imds)

%if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds, 0.8, 0.1, 0.1, 'randomized');
    save('mySets.mat','trainingSet','validationSet','testSet')
%end

layers = [
    imageInputLayer([227 227 3])
    %warstwa 1
    convolution2dLayer(11,96,'Stride',4,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    %warstwa 2
    crossChannelNormalizationLayer(5,'K',1)
    convolution2dLayer(5,256,'Stride',1,'Padding',2)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    %warstwa 3
    crossChannelNormalizationLayer(5,'K',1)
    convolution2dLayer(3,384,'Stride',1,'Padding',1)
    reluLayer
    %warstwa 4
    convolution2dLayer(3,384,'Stride',1,'Padding',1)
    reluLayer
    %warstwa 5
    convolution2dLayer(3,256,'Stride',1,'Padding',1)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    %warstwa 6
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer
    %warstwa 7
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer
    %warstwa 8
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
 ]

imageSize = [227 227 3];
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet);
augmentedValidationSet = augmentedImageDatastore(imageSize, validationSet);
augmentedTestSet = augmentedImageDatastore(imageSize, testSet);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%load 'myAlexNet.mat';
if ~exist('net', 'var')
    net = trainNetwork(augmentedTrainingSet,layers,options);
end

[YPred,scores] = classify(net,augmentedTestSet);
[S,I] = maxk(scores',5);
YValidation = testSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

top5 = sum(sum(tbl.Label(I)' == YValidation))/numel(YValidation)

function data = customreader(filename)
    load(filename,'frame');
    data=squeeze(frame(1,:,:,:));
end
