s="";
tmpFiles=imds.Files;
%tmpLabels=imds.Labels;
for i=1:length(imds.Files)
    f=tmpFiles(i);
    for j=1:length(m)
        try
            s=strrep(m(j,1),'@26.9deg_shutter in microsec\','');
            s=strrep(s,'.avi','');
            s=regexprep(s,'gain=[0-9]+\','');
            s=replace(s,'\','_');
            s=strcat(s,'\');%żeby nie dopasowywało wzorcow 2 składnikowych do 1 składnikowych
            %TODO wzorzec nie jest taki, bo zamiast ukośnika na pod folderze są
            %podkreślniki
            if contains(f,s)
                tmpLabels(i)=string(m(j,7));
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
    if ismissing(tmpLabels(i))
        throw(MException('tmpLabels(i) ismissing'))
    end
end
imds.Labels=categorical(tmpLabels);