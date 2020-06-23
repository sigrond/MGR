for i=1:length(m)
    if isnumeric(cell2mat(m(i,3)))
        m3=num2str(cell2mat(m(i,3)));
    else
        m3=m(i,3);
    end;
    if isnumeric(cell2mat(m(i,4)))
        m4=num2str(cell2mat(m(i,4)));
    else
        m4=m(i,4);
    end;
    m(i,5)=strcat(m(i,2), m3, m4)
end