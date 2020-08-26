reset(gpuDevice(1));
cz1 = fullfile(...%'minitest');
'@26.9deg_shutter in microsec');
if ~exist('imds','var')
    imds = imageDatastore(files,'LabelSource','none','IncludeSubfolders',true,'FileExtensions','.mat','ReadFcn',@(filename)customreader(filename));
    match_labels4;
    save('myImds\myImdsTM2.mat','imds')
end

tbl = countEachLabel(imds)

[YPred,scores] = classify(net,imds,'MiniBatchSize',1);
[S,I] = maxk(scores',5);
YValidation = imds.Labels;


function data = customreader(filename)
    load(filename,'frame');
    %x=randi(10);
    %y=randi(10);
    f64=frame(1:480,1:640,1);
    mf64=double(max(max(max(f64))));
    data=double(f64)./mf64;
end