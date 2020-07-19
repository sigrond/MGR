s="";
%tmpFiles=imds.Files;
%tmpLabels=imds.Labels;
%js={}
listing=dir('@26.9deg_shutter in microsec/**/*.mat');
k=1;
files={};
for i=1:length(listing)
    f=listing(i).folder;
    for j=1:length(js)
        %try
            s=strrep(js(j,1),'@26.9deg_shutter in microsec\','');
            s=strrep(s,'.avi','');
            s=regexprep(s,'gain=[0-9]+\','');
            s=replace(s,'\','_');
            %TODO wzorzec nie jest taki, bo zamiast ukoœnika na pod folderze s¹
            %podkreœlniki
            if contains(f,s)
                %tmpLabels(i)=string(m(j,5));
                files{k}=strcat(listing(i).folder,'\',listing(i).name);
                k=k+1;
                break;
            end
%         catch e
%             e.identifier
%             e.message
%             s
%             js(j,1)
%             j
%         end
    end
end
