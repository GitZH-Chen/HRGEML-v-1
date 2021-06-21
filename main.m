% Hybrid Riemannian Graph-Embedding Metric Learning for Image Set Classification
% author: Zihen Chen, one climbing the scientific Qomolangma from Mariana Trench
% date: an analyzing-math day long before love-perishing-day
% copyright@ JNU_B411
% department: the school of artificial intelligence and computer science (AI&CS)
% Contact Email:zh_chen@stu.jiangnan.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
isnormalize = false;
%% Step 1 load data
load data_VIRUS.mat
% load data_FPHA.mat
fold_num = data.fold_num; % N-fold evaluation
dr = data.para.dr;
k = data.para.k;
Train_labels = data.labels.train;
Test_labels = data.labels.test;
dataset = data.dataset;
if strcmp(dataset,'VIRUS')
    isnormalize = true;
end
accuracy_matrix = zeros(1,fold_num);
for iteration = 1:fold_num
    %% step 2 load kernel
    tic
    kmatrix_train = data.kmtraix{iteration}.Gras_train;
    kmatrix_test = data.kmtraix{iteration}.Gras_test;
    kmatrix_train_Gau = data.kmtraix{iteration}.Gauss_train;
    kmatrix_test_Gau =  data.kmtraix{iteration}.Gauss_test;
    kmatrix_train_Spd = data.kmtraix{iteration}.SPD_train;
    kmatrix_test_Spd = data.kmtraix{iteration}.SPD_test;  

   %% step 2 normalize kernel
   if isnormalize
        D_train_Gras = data.kmtraix{iteration}.D_Gras_train;
        D_test_Gras = data.kmtraix{iteration}.D_Gras_test;
        D_train_Gau = data.kmtraix{iteration}.D_Gauss_train;             
        D_test_Gau = data.kmtraix{iteration}.D_Gauss_test;
        D_train_Spd = data.kmtraix{iteration}.D_SPD_train;
        D_test_Spd = data.kmtraix{iteration}.D_SPD_test;  
        kmatrix_train = normalize(kmatrix_train,D_train_Gras,D_train_Gras);
        kmatrix_train_Spd = normalize(kmatrix_train_Spd,D_train_Spd,D_train_Spd);
        kmatrix_train_Gau = normalize(kmatrix_train_Gau,D_train_Gau,D_train_Gau);
        kmatrix_test = normalize(kmatrix_test,D_train_Gras,D_test_Gras);
        kmatrix_test_Spd = normalize(kmatrix_test_Spd,D_train_Spd,D_test_Spd);
        kmatrix_test_Gau = normalize(kmatrix_test_Gau,D_train_Gau,D_test_Gau);
   end
    [E_gras, E_Spd, E_Sgm] = Opt(kmatrix_train, kmatrix_train_Spd, kmatrix_train_Gau, Train_labels,dr,k,2);
    training_time = toc;
 %% Classification
    tic                 
    train_data_spd = zeros(dr,size(Train_labels,2));
    train_data_gras = zeros(dr,size(Train_labels,2));
    train_data_gaus = zeros(dr,size(Train_labels,2));
    test_data_spd = zeros(dr,size(Test_labels,2));
    test_data_gras = zeros(dr,size(Test_labels,2));
    test_data_gaus = zeros(dr,size(Test_labels,2));
    for i_dist = 1 : size(Train_labels,2)
        train_data_gras(:,i_dist) = E_gras' * kmatrix_train(:,i_dist);
        train_data_spd(:,i_dist) = E_Spd' * kmatrix_train_Spd(:,i_dist);
        train_data_gaus(:,i_dist) = E_Sgm' * kmatrix_train_Gau(:,i_dist);
    end
    for j_dist = 1 : size(Test_labels,2)
        test_data_gras(:,j_dist) = E_gras' * kmatrix_test(:,j_dist);
        test_data_spd(:,j_dist) = E_Spd' * kmatrix_test_Spd(:,j_dist);
        test_data_gaus(:,j_dist) = E_Sgm' * kmatrix_test_Gau(:,j_dist);
    end
    dist1 = pdist2(train_data_gras',test_data_gras','euclidean');
    dist2 = pdist2(train_data_spd',test_data_spd','euclidean');
    dist3 = pdist2(train_data_gaus',test_data_gaus','euclidean');
    dist = dist1 + dist2 + dist3;
    total = size(Test_labels,2);
    [dist_sort,index] = sort(dist,1,'ascend');
    right_num = length(find((Test_labels'-Train_labels(index(1,:))')==0));
    accuracy = right_num/total *100;
    accuracy_matrix(iteration) = accuracy;
    testing_time = toc;
    fprintf(1,'the classification accuracy of iter %d is: %.2f%% with training time: %.2fs.\n',iteration, accuracy, training_time); 
end
fprintf(1,'%d-fold average acc: %.2f%%.\n',fold_num, mean(accuracy_matrix));