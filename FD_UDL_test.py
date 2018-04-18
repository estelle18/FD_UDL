# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:22:06 2018

@author: Smbabys
"""

import numpy as np
import os
from scipy.misc import *
from sklearn.preprocessing import *
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from numpy.linalg import norm, matrix_rank
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment
from numpy.matlib import repmat
from sklearn.metrics import *

def setdata():
    n0 = 192*168
    m = 504
    C = 5
    Cperm = [5,11,21,25,34]
    
    # 每次产生相同的随机数序列（只要seed相同）
    rng = np.random.RandomState(5)
    # 降维变换矩阵
    R = rng.normal(0,1,(m,n0))
    # 正则化变换矩阵
    R = normalize(R,norm='l2').astype('float16')
#    for i in range(m):
#        R[i,:] = R[i,:]/norm(R[i,:])
    
    # 加载读取数据集
    faceimage_path = []
    filepath = 'C:\\My_Software\\Python\\learn_python_base\\lect08_codes\\FDUDL\\yale\\'
    num = 0
    
    for i in range(C):
        filename = os.path.join('%s%s%s' % (filepath, str(Cperm[i]), '\\'))
        faceimage_path.append(filename)
        
    for i in range(len(faceimage_path)):
        face_image = os.listdir(faceimage_path[i])
        num = num + len(face_image)
    
    # 创建二维数组，先定义列表，然后再转化为数组
    Data = np.array([[0 for i in range(num)] for i in range(m)])
    lable = []
    for i in range (len(faceimage_path)):
        face_image = os.listdir(faceimage_path[i])
        for j in range(len(face_image)):
            img = imread(faceimage_path[i] + face_image[j])
            img = img.reshape(n0)
            Data[:,len(lable)+j] = np.dot(R, img)
        lable.extend(np.ones(len(face_image))*(i+1))
    lable = np.array(lable).astype(int)
    # 正则化数据集
#    Data = DictNormalize(Data)
    Data = normalize(Data,norm='l2',axis=0).astype('float16')
    return Data,lable,num,R

#def DictNormalize(Dict):
#    sumDictElems = sum(abs(Dict))
#    zerosIdx = np.where(sumDictElems < 10**-15)[1]
#    Dict[:,zerosIdx] = np.random.randn(Dict.shape[0],len(zerosIdx))
#    Dict = np.dot(Dict,np.diag(1./sqrt(sum(Dict*Dict))))
   
def initialization(Data,label):
    # 子字典的原子数目
    K = 14
    # 稀疏约束
    S = 3
    C = 5
    
    total_image = Data.shape[1]
    rng = np.random.RandomState(5)
    perm = rng.permutation(total_image)
    
    # 随机初始化字典:504*14(用的是原始数据),然后正则化字典
    #     Dict = np.array([[0 for i in range(K)] for i in range(m)])  #错误
    Dict = []
    for i in range(C):
        Dict.append(Data[:,perm[i*K:(i+1)*K]])
    for i in range(C):
        Dict[i] = normalize(Dict[i],norm='l2').astype('float16')
        
    # 利用字典来聚类数据(按照论文中公式（1）来聚类)
    T = np.zeros([total_image]).astype(int)
    for t in range(total_image):
        temp = np.zeros([C,1])
        for i in range(C):
            omp = OMP(n_nonzero_coefs=S)
            omp.fit(Dict[i], Data[:,t])
            a = omp.coef_
            temp[i] = sum(abs((Data[:,t] - np.dot(Dict[i],a))))
        # np.where()[0] 表示行的索引，np.where()[1] 则表示列的索引
        pos = np.where(np.array(temp) == min(temp))
        pos_new = pos[0]
        T[t] = pos_new+1
        
    # 计算初始精度
    classes = getclass(Data,T,C)
    score,cmat = accuracy(label,T)
    return T, Dict, classes, score, cmat
'''
    根据数据的标签重新确定Data的类别
'''
def getclass(Data, T, C):
    classes = []
    for i in range(C):
        classes.append(Data[:,np.where(np.array(T)==(i+1))[0]])
    return classes

def accuracy(true_label, cluster_label):
    n = len(true_label)
#     row = [i for i in range(n)]
#     loc = true_label
#     values = [1]*n
#     cat = coo_matrix((values,(row,loc)))
#     loc_clus = cluster_label
#     cls = coo_matrix((values,(row,loc_clus))).T
#     cmat = np.dot(cls,cat).toarray()[1:6,1:6]
    
    # 利用匈牙利算法计算精度
    cmat = confusion_matrix(true_label,cluster_label).T
    row_ind, col_ind = linear_sum_assignment(-cmat)
#    score = cmat[row_ind, col_ind].sum()
#    row_ind, col_ind = linear_sum_assignment(-cmat)
#     cmat = confusion_matrix(true_label,cluster_label).T
#     row_ind = np.array([0,1,2,3,4]).astype(int)
#     col_ind = np.array([0,1,2,3,4]).astype(int)
    score = float(100*(cmat[row_ind, col_ind].sum()/n))
    return score, cmat

# def accuracy(true_label, cluster_label):
#     n = len(true_label)
#     row = [i for i in range(n)]
#     loc = true_label
#     values = [1]*n
#     cat = coo_matrix((values,(row,loc)))
#     loc_clus = cluster_label
#     cls = coo_matrix((values,(row,loc_clus))).T
#     cmat = np.dot(cls,cat).todense()[1:6,1:6]
    
#     # 利用匈牙利算法计算精度
# #    row_ind, col_ind = linear_sum_assignment(cmat)
# #    score = cmat[row_ind, col_ind].sum()
#     row_ind, col_ind = linear_sum_assignment(-cmat)
#     score = 100*(cmat[row_ind, col_ind].sum()/n)
#     return score,cmat


'''
    计算Fisher及重构误差,估计算法的性能
'''
def showEnergy(Dict, classes, params):
    S = params['S']
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    C = len(Dict)
    n = Dict[0].shape[0]
    K = params['K']
    
    # 计算重构误差及fisher项的熵
    energy_re = 0
    energy_d1 = 0
    
    for i  in range(C):
        omp = OMP(n_nonzero_coefs=S)
        omp.fit(Dict[i], classes[i])
        A_coef = omp.coef_
        err = classes[i] - np.dot(Dict[i],A_coef.T)
        energy_re = energy_re + sum(sum(err**2))
        
    # 计算fisher类内熵
    m = np.zeros([n,C])
    for i in range(C):
        m[:,i] = np.mean(Dict[i],axis=1) #504*1
        err = Dict[i] - repmat(m[:,i].reshape(n,1),1,K)
        energy_d1 = energy_d1 + sum(sum(err**2))
        
    # 计算fisher类间熵
    mc = np.mean(m,1)
    err = repmat(mc.reshape(n,1),1,C) - m
    energy_d2 = K*sum(sum(err**2))
    energy_d = alpha1*energy_d1 - alpha2*energy_d2
    
    energy = energy_re + energy_d
    values = [energy_re,energy_d1,energy_d2, energy_d]
    return energy,values
    
'''
    字典学习（Fisher鉴别算法）
'''
def FDFaces(classes, params):
    C = len(classes)
    K = params['K']
    S = params['S']
    n = classes[1].shape[0]
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    numIteration = params['numIteration']
    
    # 正则化dictionary及data
#    for i in range(C):
#        normalize_model = Normalizer()
#        normalize_model.fit(classes[i].T.astype(float))
#        classes[i] = normalize_model.transform(classes[i])
    for i in range(C):
        classes[i] = normalize(classes[i],norm='l2',axis=0).astype('float16')
    
    D = []
    # 初始化字典
    for i in range(C):        
        if classes[i].shape[1]<K:
            rng = np.random.RandomState(5)
            perm = rng.permutation(classes[0][:,:2].shape[1])
            add_num = K - classes[0][:,:2].shape[1]
            col = []
            col_first = [k for k in np.arange(classes[0][:,:2].shape[1])]
            col.extend(col_first)
            col_last = [j for j in perm[:add_num]]
            col.extend(col_last)
            if len(col)<(K/2):
                classes_new = np.zeros([504,100])
                a = (K-len(col))
                for  j in range(a):
                    index = 2*classes[0][:,:2].shape[1]
                    classes_new[:,j*index:(j+1)*index] = np.column_stack((classes[0][:,:2],classes[0][:,:2]))
                D.append(classes_new[:,:K])
#                print('classes_new长度：',classes_new.shape[1])
#             rng = np.random.RandomState(5)
#             perm = rng.permutation(classes[i].shape[1])
#             add_num = K - classes[i].shape[1]
#             col = []
#             col_first = [k for k in np.arange(classes[i].shape[1])]
#             col.extend(col_first)
#             col_last = [j for j in perm[:add_num]]
#             col.extend(col_last)
#             if len(col)<K:
#                 for j in range(K-len(col)):
#                     for z in classes[i][:,j]:
#                         classes_new = np.column_stack(classes[i],classes[i][:,z])
#                 print('col长度：',len(col))
            else:
                D.append(classes[i][:,col])
        else:
            D.append(classes[i][:,:K])
    
    # D 的解析解
    finished = 0
    
    """
        类内推导公式、类间推导公式、重构推导公式
    """
    P1 = np.zeros([C*K,C*K])
    P0 = np.zeros([K,K])
    q = np.ones([K,1])/K
    for i in range(K):
        p = np.zeros([K,1])
        p[i] = 1
        P0 = P0 + np.dot((p-q),(p-q).T)
    for i in range(C):
        P1[i*K:(i+1)*K,i*K:(i+1)*K] = P0
    
    P2 = np.zeros([C*K,C*K])
    s = np.ones([C*K,1])/(C*K)
    for i in range(C):
        r = np.zeros([(C*K),1])
        r[i*K:(i+1)*K,0] = 1
        P2 = P2 + np.dot((r-s),(r-s).T)
        P2 = P2*K
        
    Fish = alpha1*P1 - alpha2*P2
    
    for iterNum in range(numIteration):
        # 按照初始字典来求稀疏系数矩阵
        A_coeff = []
        for i in range(C):
            omp = OMP(n_nonzero_coefs=S)
            omp.fit(D[i],classes[i])
            coeff = omp.coef_.reshape(classes[i].shape[1],K)
            A_coeff.append(coeff)
#            print('========',D[i].shape,classes[i].shape,coeff.shape,'======\n')
        
        Q = np.zeros([n,C*K])
        for i in range(C):
            Q_dot = np.dot(classes[i],A_coeff[i])
#            print(Q_dot.shape,classes[i].shape,A_coeff[i].shape)
            Q[:,i*K:(i+1)*K] = Q_dot # 后期可以查看这里的系数矩阵是否需要转置(经测试不需要转置)
        A = np.zeros([C*K,C*K])
        for i in range(C):
            A[i*K:(i+1)*K,i*K:(i+1)*K] = np.dot(A_coeff[i].T,A_coeff[i])
        W = A + Fish
        
        if np.linalg.det(W) == 0:
            D0 = np.dot(Q,np.linalg.pinv(W))
        else:
            D0 = np.dot(Q,np.linalg.inv(W))
        D0[np.where(np.isnan(D0)==1)] = 0
        D0[np.where(np.isinf(D0)==1)] = 10**20
        D0 = normalize(D0,norm='l2', axis=0).astype('float16')
#        print('D0:504*70',D0.shape)
        
        for i in range(C):
            D[i] = D0[:,i*K:(i+1)*K]
#            print('D0分解为D[i]：','D[',i,']形状：',D[i].shape)
#        print('\nD0分解为D[i]：','D长度：',len(D))
            
        # 至此字典更新完毕
        
        # 求更新字典后，进行聚类的精度
        first = 0
        last,_ = showEnergy(D, classes, params)
        if (abs(last-first)/last) < 0.002:
            if finished<1:
                finished += 1
            else:
                print(str(iterNum))
                return
        else:
            finished = 0
            
        first = last
    
    return D


'''
    信号分配
    Input  : Dict : dictionary 
            x : a piece of data 
            S : sparse level constraint
    Output : k : cluster label of data
    
'''
def cluster_result(Dict, x, params):
    C = len(Dict)
    S = params['S']
    temp = np.zeros([C,1])
    for i in range(C):
        omp = OMP(n_nonzero_coefs=S)
        omp.fit(Dict[i], x)
        a = omp.coef_
        temp[i] = sum((x - np.dot(Dict[i],a)))
    # np.where()[0] 表示行的索引，np.where()[1] 则表示列的索引
    pos = np.where(np.array(temp) == min(temp))[0]
    k = pos[0]+1
    return k

'''
    交替迭代进行人脸分配和字典学习
'''
def loopFace():
    
    Data, label,num = setdata()
    T, Dict, classes, score, cmat = initialization(Data,label)
    
    # iteration: the number of iterations for Alternately Doing Signal Assignment and Dictionary Learning
    iteration =80
    C = 5
    # numIteration: the number of iterations for Dictionary Learning
    params = dict(alpha1=0.2,alpha2=0, S=3, K=14, numIteration=80)    
    
    first,_ = showEnergy(Dict,classes,params)
    energy = np.zeros([iteration,1])
    values = np.zeros([iteration,4])
    acc = np.zeros([iteration,1])
    
    finished = 0
    locked = 0
    
    for i in range(iteration):
        
        # 1. 字典学习阶段
        Dict_new = FDFaces(classes, params)
        print()
        # 2. 信号分配
        T = np.zeros([num, 1])
        for j in range(num):
            T[j] = cluster_result(Dict_new, Data[:,j],params)
        classes = getclass(Data,T,C)
        
        for k in range(C):
            dimension = classes[k].shape[1]
            if dimension == 0:
                if k==(C-1):
                    classes[k] = classes[k-1][:,0].reshape(504,1)
                    classes[k-1] = classes[k-1][:,1:-1]    
                classes[k] = classes[k+1][:,0].reshape(504,1)
                classes[k+1] = classes[k+1][:,1:-1] 
        
        # 3. 计算聚类准确率
        acc[i], _ = accuracy(label,T)
        print('\n iteration {0:d}, alpha1 = {1:.2f}, alpha2 = {2:.2f}.\n'.format(i, params['alpha1'], params['alpha2']))
        print('accuracy = {0:.4f}\n'.format(acc[i,0]))
        energy[i], values[i,:] = showEnergy(Dict_new,classes, params)
        print('energy_r = {0:.4f}, energy_d1 = {1:.4f}, energy_d2 = {2:.4f}\n'.format(values[i,0],values[i,1],values[i,2]))
        print('energy_r = {0:.4f}, energy_d ={1:.4f}\n'.format(values[i,0],values[i,3]))
        print('total energy = {0:.4f}\n'.format(energy[i,0]))
        
        last = energy[i]
        if (abs((last-first)/last)) < 0.004:
            if finished < 2:
                finished += 2
            else:
                if params['alpha2'] < 28:
                    params['alpha1'] += 0.4
                    params['alpha2'] += 2
                    finished = 0
                else:
                    return
        elif (abs((last-first)/last) < 0.04):
            finished = 0
            if(locked < 3):
                locked = locked + 1;
            else:
                if params['alpha2'] < 28:
                    params['alpha1'] += 0.4
                    params['alpha2'] += 2
                    locked = 0
                else:
                    return
        else:
            locked = 0
            finished = 0
            
        first = last
    
    
if __name__ == '__main__': 
    Data, label, num, R = setdata()
    T, Dict, classes, score, cmat = initialization(Data,label)
#    loopFace()
