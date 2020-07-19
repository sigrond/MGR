%test 7 - nauczenie sieci klas ró¿nych substancji i ich stê¿eñ
%bez utraty danych przez skalowanie lub strukturê sieci
%wyko¿ystanie ca³ych klatek
reset(gpuDevice(1));
cz1 = fullfile(...%'minitest');
'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(files,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels3;
    save('myImds\myImds7g23.mat','imds')
end

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds,0.8, 0.1, 0.1 ... 
        ...%399,49,49 ...
         ...%100,10,10 ...
        ,'randomized');
    save('mySets\mySets7g23.mat','trainingSet','validationSet','testSet')
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
    'none','Name','input')
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0,'Name','conv1')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu1')
    maxPooling2dLayer([3 3],'Stride',2,'Name','MP1')
    convolution2dLayer([3 3],64,'Stride',[2 2],'Padding',1,'Name','conv2')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu2')
    %maxPooling2dLayer([3 3],'Stride',2,'Name','MP2')
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',1,'Name','conv3')
    additionLayer(2,'Name','add1')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu3')
    maxPooling2dLayer([3 3],'Stride',2,'Name','MP3')
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',1,'Name','conv4')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu4')
    %maxPooling2dLayer([3 3],'Stride',2,'Name','MP4')
    convolution2dLayer([3 3],64,'Stride',[2 2],'Padding',1,'Name','conv5')
    additionLayer(2,'Name','add2')
    batchNormalizationLayer('Name','BN5')
    reluLayer('Name','relu5')
    maxPooling2dLayer([3 3],'Stride',2,'Name','MP5')
    convolution2dLayer([3 3],64,'Stride',[1 1],'Padding',0,'Name','conv6')
    batchNormalizationLayer('Name','BN6')
    reluLayer('Name','relu6')
    maxPooling2dLayer([3 3],'Stride',2,'Name','MP6')
    %warstwa 6
    %fullyConnectedLayer(500)
    fullyConnectedLayer(512,'Name','fc1')
    %net7d7.Layers(9)
    %fc1
    batchNormalizationLayer('Name','BN7')
    reluLayer('Name','relu7')
    %crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25,'Name','drop1')
    %warstwa 7
    fullyConnectedLayer(256,'Name','fc2')
    %net7d7.Layers(13)
    %fc2
    reluLayer('Name','relu8')
    %crossChannelNormalizationLayer(3,'K',2)
    dropoutLayer(0.25,'Name','drop2')
    %warstwa 8
    fullyConnectedLayer(53,'Name','fc3')
    %net7c15.Layers(17)
    %fc3
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classif')
 ]

lgraph = layerGraph(layers);
skipConv1 = convolution2dLayer(1,64,'Stride',2,'Name','skipConv1');
lgraph = addLayers(lgraph,skipConv1);
lgraph = connectLayers(lgraph,'MP1','skipConv1');
lgraph = connectLayers(lgraph,'skipConv1','add1/in2')

skipConv2 = convolution2dLayer(1,64,'Stride',2,'Name','skipConv2');
lgraph = addLayers(lgraph,skipConv2);
lgraph = connectLayers(lgraph,'MP3','skipConv2');
lgraph = connectLayers(lgraph,'skipConv2','add2/in2')

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
    'LearnRateDropPeriod',3,...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'MiniBatchSize', 8, ...
    'CheckpointPath', 't7g');

%load 'myAlexNet.mat';
if ~exist('net', 'var')
    net = trainNetwork(trainingSet,lgraph,options);
else
    net = trainNetwork(trainingSet,layerGraph(net),options);
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
