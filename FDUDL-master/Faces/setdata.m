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

% matlab���õİ��������� rng�������������ʼֵ��0����������ӣ������ǷǸ���������'v5normal'�����ɷ�ʽ��
% ����help���˵��Ӧ���Ǵ�ͳMATLAB5.0��̬���� ʹ����������֮��ÿ�����ɵ�������������ι̶���
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

