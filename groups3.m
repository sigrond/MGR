for i=1:length(m)
    if isnumeric(cell2mat(m(i,5)))
        m5=num2str(cell2mat(m(i,5)));
    else
        m5=m(i,5);
    end;
    if isnumeric(cell2mat(m(i,6)))
        m6=num2str(cell2mat(m(i,6)));
    else
        m6=m(i,6);
    end;
    m(i,7)=strcat(m(i,2), m5, 'r', m6)
end