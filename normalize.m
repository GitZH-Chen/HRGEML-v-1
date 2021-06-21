% author: Zihen Chen
% Contact Email:zh_chen@stu.jiangnan.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kmatrix] = normalize(K,D_trian,D_test)
if isempty(D_test)
    D_test = D_trian;
end
kmatrix = D_trian * K * D_test;
end