% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.
function G = generate_Graphs(weights, neighbor_index,Train_lables,mute)
D = size(weights,1); % Sets numbers
num_w = 0; 
num_b = 0;
%% calculate LLE weights
if mute~=1
    fprintf('-->Generating graph\n');
end
%Within Graph
G_w = zeros(D);
for tmpC1 = 1:D
    tmpIndex = find(Train_lables == Train_lables(tmpC1));
    tmpJudge = ismember(neighbor_index(:,tmpC1),tmpIndex);
    num_w = num_w + sum(tmpJudge);
    tmpWeightsIndx = neighbor_index(tmpJudge,tmpC1);
    G_w(tmpC1,tmpWeightsIndx) = weights(tmpC1,tmpWeightsIndx);
end
%Between Graph
G_b = zeros(D);
 for tmpC2 = 1:D
    tmpIndex = find(Train_lables ~= Train_lables(tmpC2));
    tmpJudge = ismember(neighbor_index(:,tmpC2),tmpIndex);
    num_b = num_b + sum(tmpJudge);
    tmpWeightsIndx = neighbor_index(tmpJudge,tmpC2);
    G_b(tmpC2,tmpWeightsIndx) = weights(tmpC2,tmpWeightsIndx);
 end
G = G_w/num_w - G_b/num_b;
%G = G_w - G_b;
if mute~=1
    fprintf('Done.\n');
end






