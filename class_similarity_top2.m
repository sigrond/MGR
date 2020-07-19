% klasy podobne
labels=tbl.Label;
%ntbl = countEachLabel(imds)
nLabels=ntbl.Label;%z imds jak dla sieci
%p={};
G=digraph;
for i=1:length(labels)
    G=addnode(G,char(labels(i)));
end
for i=1:length(labels)
    M=containers.Map;
    Mg=containers.Map;
    for j=1:length(YValidation)
        if YValidation(j)==labels(i)
            %for k=1:size(I,1)
                if isKey(M,char(YPred(j)))   
                    M(char(YPred(j)))=M(char(YPred(j)))+1;
                else
                    M(char(YPred(j)))=1;
                end
                [max2, I2]=maxk(scores(j,:),2);
                m2=I2(2);
                if isKey(Mg,char(nLabels(m2)))   
                    Mg(char(nLabels(m2)))=Mg(char(nLabels(m2)))+1;
                else
                    Mg(char(nLabels(m2)))=1;
                end
            %end
        end
    end
    p(i)={M};
    keys=M.keys;
    values=cell2mat(M.values);
    [B,ind]=maxk(values,2);
    b={[0 1 0]};
    cb(1:length(ind))=b;
    newEdges = table(values(ind)', cb', 'VariableNames',{'Weight','Color'});
    G=addedge(G,char(labels(i)),keys(ind),newEdges);
    
    pg(i)={Mg};
    keys=Mg.keys;
    values=cell2mat(Mg.values);
    [B,ind]=maxk(values,2);
    g={[0 0 1]};
    cg(1:length(ind))=g;
    newEdges = table(values(ind)', cg', 'VariableNames',{'Weight','Color'});
    G=addedge(G,char(labels(i)),keys(ind),newEdges);
end
%H=simplify(G);
H=G;
LWidths = 10*H.Edges.Weight/max(H.Edges.Weight);
colors=cell2mat(H.Edges.Color);
%plot(H,'EdgeLabel',H.Edges.Weight,'LineWidth',LWidths,'Layout','force','WeightEffect','inverse')
plot(H,'EdgeLabel',H.Edges.Weight,'LineWidth',LWidths,'Layout','layered','EdgeColor',colors)