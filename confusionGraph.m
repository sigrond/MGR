% klasy podobne
labels=tbl.Label;
%p={};
G=digraph;
for i=1:length(labels)
    G=addnode(G,char(labels(i)));
end
for i=1:length(labels)
    M=containers.Map;
    for j=1:length(YValidation)
        if YValidation(j)==labels(i)
            for k=1:size(scores,2)
                if isKey(M,char(labels(k)))   
                    M(char(labels(k)))=M(char(labels(k)))+scores(j,i);
                else
                    M(char(labels(k)))=scores(j,i);
                end
            end
        end
    end
    p(i)={M};
    keys=M.keys;
    values=cell2mat(M.values);
    [B,ind]=maxk(values,5);
    G=addedge(G,char(labels(i)),keys(ind),values(ind));
end
%H=simplify(G);
H=G;
LWidths = 10*H.Edges.Weight/max(H.Edges.Weight);
plot(H,'EdgeLabel',H.Edges.Weight,'LineWidth',LWidths,'Layout','force','WeightEffect','inverse')
