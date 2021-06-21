% author: Zihen Chen
% Contact Email:zh_chen@stu.jiangnan.edu.cn
% ADMM for Simplex-Constrained LLE weights

% [W] = ADMM(k,N,Q,B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W] = ADMM(k,N,Q,B)
W = zeros(k*N,1);
I = eye(k*N);
one = ones(N,1);
num = size(B,1);
null = zeros(num,num);
tau = 2;
mu =10;

%% initialization p,W,u
p =10;
u = W;
Z = W;
Z_last = [];
tol   = 1e-3;
iter    = 1;
terminate = false;

while  ( ~terminate )
    %% update W
    A = [Q+p*I,B';B,null];
    b =[p*(Z -u);one];
    opts.SYM = true;
    temp = linsolve(A,b,opts);
    %temp = A\b;
    W = temp(1:k*N,:);
    %% update Z
    Z_last = Z;
    temp2 = W + u;
    temp3 = temp2;
    temp3(find(temp3 <= 0)) = 0;
    Z = temp3;
    %% updat multipliers
    u = u + W - Z;    
    %% computing errors
    r = W - Z;
    s = p * (Z_last -Z);
    err_r = norm(r);
    err_s = norm(s);
    if ( err_r <= tol && err_s <= tol)
        terminate = true;
        %fprintf('err of r: %2.4f, err of s: %2.4f, iter: %3.0f \n',err_r, err_s, iter);
    else
        if (mod(iter, 25)==0)
            terminate = true;
            %fprintf('reach max iter \n');
            %fprintf('err of r: %2.4f, err of s: %2.4f, iter: %3.0f \n',err_r, err_s, iter);
        end
    end
    %% update p
    if( err_r > mu * err_s)
        p =tau*p;
        u = u/tau;
    else if(err_s > mu * err_r)
        p = p/tau;
        u = u * tau;
        end
    end
    %% next iteration number
    iter = iter + 1;
end
end