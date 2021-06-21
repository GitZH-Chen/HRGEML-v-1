% author: Zihen Chen
% Contact Email:zh_chen@stu.jiangnan.edu.cn
% Optimization algorithm on graph embedding and metric learning 

% [U_gras, U_Spd, U_Sgm] = Opt( k_Gras, k_Spd, k_Sgm, Train_lables,dr,k,mute)
% mute : if mute = 1, then mute all fprintf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U_gras, U_Spd, U_Sgm] = Opt( k_Gras, k_Spd, k_Sgm, Train_lables,dr,k,mute)
D = size(k_Gras,1); % Sets numbers
K_c = {k_Gras,k_Spd,k_Sgm};
U = cell(1,3);
for q = 1:3
    if mute ~= 1
        fprintf('For %dth embedding:\n',q);
    end
    kernelMatrix = K_c{q};
    %% calculate LLE weights
    [w,neighbor_index] = simplex_lle_weights_ADMM(kernelMatrix,k,mute);
    weights = zeros(D); 
    %get 40*40 weights matrix
    for ii = 1:D
        weights(ii,neighbor_index(:,ii)) = w(:,ii); % transpose and insert to 40 * 40
    end    
    %% calculate qth Graph
    G = generate_Graphs(weights,neighbor_index, Train_lables,mute);
    %% calculate Fq
    F = zeros(D);
    for ii = 1:D
        temp_G = G(ii,:);
        G_diag = diag(temp_G);
        tmpkernel = repmat(kernelMatrix(:,ii),1,D) - kernelMatrix;
        F  = F + tmpkernel * G_diag * tmpkernel' ;
    end
    F=(F+F')/2;
    [V,eigen] = eig(F);
    [eigen_sort,index] = sort(diag(eigen),'ascend');
    V_sort = V(:,index);
    U{q} = V_sort(:,1:dr);
end
U_gras = U{1}; 
U_Spd = U{2};
U_Sgm = U{3};
if mute~=1
    fprintf('Done.\n');
end
