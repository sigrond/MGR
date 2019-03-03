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
        f1=squeeze(frame);
        for k=1:24
            f2=f1((k-1)*20+1:k*20,:,:);
            m=double(max(max(max(f2))));
            if m>0
                s=uint16(floor(65545/m));
                strip=f2.*s;
                save(strcat(list(i).folder,'\',strrep(list(i).name,'.avi',''),'\',int2str(j),'s',int2str(k),'.mat'),'strip');
            end
        end
    end
end