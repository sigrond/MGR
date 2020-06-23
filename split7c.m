list = dir('@26.9deg_shutter in microsec/*/*.avi');
for i=1:length(list)
    filename=strcat(list(i).folder,'\',list(i).name);
    ai=aviinfo(filename);
    n=ai.NumFrames;
    foldername=strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''));
    foldername=strrep(foldername,'H20','H2O');
    if ~exist(foldername,'dir')
        mkdir(foldername);
    end
    strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat')
    if ~exist(strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat'),'file')
        na=1;
        nb=200;
        while nb<=n
            if nb>n
                nb=n;
            end
            if na>nb
                break;
            end
            while 1
                try
                    frames=AviReadPike_Split(filename,na:nb);
                    break;
                catch exception
                    nb=nb-1;
                    memory
                    if na>nb
                        throw(exception)
                    end
                end
            end
            sumedframes=squeeze(sum(frames,1));
            %œrednia klatka dla filmu
            averageframe=sumedframes(:,:,2:3)./(nb-na);
            for j=na:nb
                tframe=double(squeeze(frames(j-na+1,:,:,2:3)));
                %odjêcie t³a filmu
                fframe=tframe-averageframe;
                minframepixel=min(min(min(tframe)));
                if minframepixel < 0
                    fframe=fframe-minframepixel;
                end
                m=double(max(max(max(fframe))));
                if m>0
                    s=65545/m;
                    frame=uint16(fframe.*s);
                    save(strrep(strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(j),'.mat'),'H20','H2O'),'frame');
                end
            end
            na=nb+1;
            nb=na+200;
            if nb>n
                nb=n;
            end
        end
    else
        strcat(list(i).folder,'_',strrep(list(i).name,'.avi',''),'\',int2str(n-10),'.mat')
    end
end