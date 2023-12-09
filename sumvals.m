function edge_sum = sumvals(d)
e = entries(d,'struct');
edge_sum = 0;
for ie = 1:length(e)
    cj = e(ie).Key;
    val = d(cj);
    edge_sum = edge_sum + val;
end
%fprintf('edge_sum = %d\n',edge_sum);