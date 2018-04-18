% =========================================================================
%                Loading the YaleB DataSet
% =========================================================================
clc,clear
% n0 -size of face images
n0 = 192 * 168;
% n -number of face features 
n = 504;
C = 5;

% select 5 specific people from yaleB
Cperm = [34 21 5 25 11];

% matlab内置的帮助很有用 rng是生成随机数初始值，0是随机数种子（必须是非负整数），'v5normal'是生成方式，
% 按照help里的说明应该是传统MATLAB5.0正态生成 使用这个命令后，之后每次生成的随机数都是依次固定的
rng(5,'v5uniform');
% randomly initialize R 
R = randn(n,n0);
% normalize R
for i = 1:n
    R(i,:) = R(i,:)/norm(R(i,:));
end

Data = [];
label = [];

file_path = 'C:\Users\estella_shu\Documents\MATLAB\FDUDL-master\Faces\yale\';
for i = 1:C
    % load face images
    file = dir(strcat(file_path, num2str(Cperm(i)),'\','\*.jpg'));
    p = size(file, 1);
    for j = 1:p
        pic = imread(strcat(file_path, num2str(Cperm(i)),'\',num2str(j),'.jpg'));
        pic = double(pic(:));
        Data(:,length(label)+j) = R * pic;                     % use R to reduce dimentions 
    end
    label = [label;ones(p,1)*i];    
end
    
Data = DictNormalize(Data);

