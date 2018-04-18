function D = FDFaces(class, params)
% =========================================================================
%                          Fisher Discriminant algorithm
% =========================================================================
% INPUT ARGUMENTS:
% class				a cell where each element is a cluster for a class
%                                                
% params            
%    K, ...         the number of atoms in a sub-dictionary              
%    S, ...         sparse level constraint
%    theta1, ...         the trade-off factor between reconstruction and
%                                within-class distermination.
%    theta2, ...         the trade-off factor between reconstruction and
%                                between-class distermination.
%    numIteration, ...      the number of iterations
% =========================================================================
% OUTPUT ARGUMENTS:
%  D                a cell where each element is a sub-dictionary for a class
% =========================================================================

C = length(class);
K = params.K;
S = params.S;
n = size(class{1}, 1);
theta1 = params.theta1;
theta2 = params.theta2;
numIteration = params.numIteration;

D = cell(1,C);
%initialize the dictionary.
for i = 1:C
	D{i} = class{i}(:,1:K);
end
%normalize the dictionary.
for i = 1:C
	D{i} = DictNormalize(D{i});
end    

finished = 0;  
aaa = 0;
%%%%%%%

    % 类内推导
    P1 = zeros(C*K,C*K); %P
    temp = zeros(K,K); %P0
    q = ones(K,1)/K;
    for i = 1:K
        p = zeros(K,1);
        p(i) = 1;
        temp = temp + (p - q) * (p - q)';  %P0
    end
    for i = 1:C
        P1((i-1)*K+1:i*K,(i-1)*K+1:i*K) = temp; %P
    end   
    
    % 类间推导
    P2 = zeros(C*K,C*K);  %R
    s = ones(K*C,1)/C;
    for i = 1:C
        r = zeros(C*K,1);
        r((i-1)*K+1:i*K) = 1;         
        P2 = P2 + (r - s) * (r - s)'; %R
    end
    
    T = theta1*P1 - theta2*P2;  %Fisher (T = theta1*P1 - P2 )
%%%%%%%%

for iterNum = 1:numIteration
    Alpha = cell(C,1); %稀疏系数
    for i = 1:C
        Alpha{i} = OMP(D{i}, class{i}, S); 
        fprintf('\n D: (%d,%d), class: (%d,%d), coeff: (%d,%d).\n', size(D{i},1),size(D{i},2),size(class{i},1),size(class{i},2),size(Alpha{i},1),size(Alpha{i},2));
    end
    
	Q = zeros(n,C*K);   %504*（5*14）
    for i = 1:C
        Q(:,(i-1)*K+1:i*K) = class{i}*(Alpha{i})';
    end
    
    A = zeros(C*K,C*K);
    for i = 1:C
        A((i-1)*K+1:i*K,(i-1)*K+1:i*K) = Alpha{i}*(Alpha{i})';
    end
    

    W = A + T;  
    if(rank(W))
        D0 = Q / W;	    
    else
        D0 = Q * pinv(W);
    end
	D0(isnan(D0)==1) = 0;
	D0(isinf(D0)==1) = 1e20;
    D0 = DictNormalize(D0);    
	
    
    for i = 1:C
        D{i} = D0(:,(i-1)*K+1:i*K);
    end    
  
    
    bbb = showEnergy( D, class, S, theta1, theta2 );           
    if(abs(bbb-aaa)/bbb < 0.002)
		if(finished < 1)
			finished = finished + 1;			
        else		
             fprintf(2, num2str(iterNum));  % 输出字典迭代次数
             return 
        end    
    else		
        finished = 0;  
    end    
    aaa = bbb;          
    
end









