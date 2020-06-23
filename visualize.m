%layer = 5;
layer = 2;
%channels = 1:128;
channels = 1:64;

I = deepDreamImage(net,'conv6',channels, ...
    'PyramidLevels',1, ...
    'Verbose',0, ...
    'InitialImage',img(1:128,1:128));

figure
for i = channels
    %subplot(12,12,i)
    subplot(8,8,i)
    imshow(I(:,:,:,i))
end