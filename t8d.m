%test 7 - nauczenie sieci klas ró¿nych substancji i ich stê¿eñ
%bez utraty danych przez skalowanie lub strukturê sieci
%wyko¿ystanie ca³ych klatek
%reset(gpuDevice(1));
cz1 = fullfile(...%'minitest');
'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(cz1,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels3;
    save('myImds8.mat','imds')
end

tbl = countEachLabel(imds)

if ~exist('trainingSet','var') || ~exist('validationSet','var') || ~exist('testSet','var')
    [trainingSet,validationSet, testSet] = splitEachLabel(imds,...%0.8, 0.1, 0.1 ...
        ...%399,49,49 
         399,49,49 ...
        ,'randomized');
    save('mySets8.mat','trainingSet','validationSet','testSet')
end

img = readimage(trainingSet, 206);

% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    %img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
% Get labels for each image.
trainingLabels = trainingSet.Labels;

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels,'Learners',templateKernel('BoxConstraint',100));

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet1(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

%helperDisplayConfusionMatrix1(confMat)

YValidation = testSet.Labels;
accuracy = sum(predictedLabels == YValidation)/numel(YValidation)

function data = customreader(filename)
    load(filename,'frame');
    data=int16(frame(1:64,1:64,1));
end