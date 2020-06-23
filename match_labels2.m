s="";
tmpFiles=imds.Files;
tmpLabels=imds.Labels;
for i=1:length(imds.Files)
    f=tmpFiles(i);
    for j=1:length(m)
        try
            s=strrep(m(j,1),'@26.9deg_shutter in microsec\','');
            s=strrep(s,'.avi','');
            s=regexprep(s,'gain=[0-9]+\','');
            s=replace(s,'\','_');
            %TODO wzorzec nie jest taki, bo zamiast ukoœnika na pod folderze s¹
            %podkreœlniki
            if contains(f,s)
                tmpLabels(i)=m(j,5);
                break;
            end
        catch e
            e.identifier
            e.message
            s
            m(j,1)
            j
        end
    end
end
imds.Labels=tmpLabels;