list = dir('@26.9deg_shutter in microsec/*/*.avi');
foldernames=strings(length(list),1);
filenames=strings(length(list),1);
for i=1:length(list)
    filenames(i)=strcat(strrep(list(i).folder,strcat(pwd,'\'),''),'\',list(i).name);
    foldernames(i)=strcat(strrep(list(i).folder,strcat(pwd,'\'),''),'_',strrep(list(i).name,'.avi',''));
end
