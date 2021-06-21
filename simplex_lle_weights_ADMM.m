% author: Zihen Chen
% Contact Email:zh_chen@stu.jiangnan.edu.cn
% Simplex-Constrained LLE weights (using K nearest neighbors)

% [W,index] = lle_weights(X,K)
% X    : Kernel matrix N * N
% K    : number of neighbors
% W    : embedding weights
% index: nearsest neighbors index 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W,neighborhood] = simplex_lle_weights_ADMM(X,k,mute)
N = length(X);
if mute~=1
    fprintf(1,'KLLE running on %d Sets\n',N);
end
%% Step1: compute pairwise distances & find neighbour
if mute~=1
    fprintf(1,'-->Finding %d nearest neighbours in Hillbert space.\n',k);
end
X2 = diag(X,0)';
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X;
[~,index] = sort(distance);
neighborhood = index(2:(1+k),:);
%% Step2: solve for reconstruction weights
if mute~=1
    fprintf(1,'-->Solving for reconstruction weights.\n');
end
temp = ones(1,k);
W = zeros(k*N,1);
% construct Q & B
for ii=1:N
   q_i = neighborhood(:,ii);                        %neighborhood of ii
   C = repmat(X(ii,ii),k,k) + X(q_i,q_i) - repmat(X(ii,q_i),k,1) - repmat(X(q_i,ii),1,k);   % Xi^T * Xi     
   temp_w = ADMM(k,1,2*C,temp);
   W((k*(ii-1)+1):k*ii) = temp_w;
end
W = reshape(W,[k,N]);
end

