function helperDisplayConfusionMatrix1(confMat,tbl)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

%digits = '0':'9';
%colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-30s',1,11);
header = sprintf(format,'class                       |',tbl{:,1});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:size(tbl,1)
    fprintf('%-30s',   [tbl{idx,1} '      |']);
    fprintf('%-30.2f', confMat(idx,:));
    fprintf('\n')
end
end