list = dir('@26.9deg_shutter in microsec/*/*.avi');
for i=1:length(list)
    filename=strcat(list(i).folder,'\',list(i).name);
    ai=aviinfo(filename);
    n=ai.NumFrames;
    foldername=strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''));
    if ~exist(foldername,'dir')
        mkdir(foldername);
    end
    strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat')
    if ~exist(strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat'),'file')
        for j=1:n
            frame=AviReadPike_Split(filename,j);
            f1=squeeze(frame);
            f2=f1;
            m=double(max(max(max(f2))));
            if m>0
                s=uint16(floor(65545/m));
                strip=f2.*s;
                save(strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(j),'.mat'),'frame');
            end
        end
    else
        strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat')
    end
end