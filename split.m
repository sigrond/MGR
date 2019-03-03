list = dir('cz1/*.avi');
for i=1:length(list)
    filename=strcat(list(i).folder,'\',list(i).name);
    ai=aviinfo(filename);
    n=ai.NumFrames;
    foldername=strcat(list(i).folder,'\',strrep(list(i).name,'.avi',''));
    if ~exist(foldername,'dir')
        mkdir(foldername);
    end
    for j=1:n
        frame=AviReadPike_Split(filename,j);
        save(strcat(list(i).folder,'\',strrep(list(i).name,'.avi',''),'\',int2str(j),'.mat'),'frame');
    end
end