#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:59:52 2018

@author: ning
"""

import numpy as np
import os

from numpy.random import seed
from imageio import imread
from sklearn.preprocessing import normalize

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
    
WIDTH_OF_IMAGE = 192
HEIGHT_OF_IMAGE = 168

size_of_faceImage = WIDTH_OF_IMAGE * HEIGHT_OF_IMAGE

FEATURES_OF_FACE = 504

C = 5 # COUNT
K = 14 # NUMBER_OF_ATOMS_IN_SUB_DICTIONARY
S = 4 # SPARSE_LEVEL_CONSTRAINT =

seed(4)

# 13  69
#
class params:
    """
    param used
    """
    theta1 = 0
    theta2 = 0
    S = 4
    K = 14
    numIteration = 80
    

########################################
#        """ 
#        preprocessing methods
#        include
#            - normalize method
#            - read picture as ndarray methods
#            - R generation
#        """
########################################

def dict_normalization(data):
    """
        normalize dict data
        Args:
            data: data to be normalized
        return:
            data: normalized data
    """
    return normalize(data, axis = 0)

def generate_R():
    """
        generate size
        return:
            R: who the fucking hell know what's this
                shape is 504 * 32256 
    """
    R = np.random.randn(FEATURES_OF_FACE, WIDTH_OF_IMAGE * HEIGHT_OF_IMAGE)

    R = dict_normalization(R)
    return R


def read_pic_as_dict():
    """
        read all file and generate data 
        Args:
            
        return:
            data: all img file data 504 * 316
            label: true label. size is 316
            R: R
    """
    label = []
    
    label_dic = {
        '34': 0,
        '21': 1,
        '5': 2,
        '25': 3,
        '11': 4
        }
    
    R = generate_R()
    
    img_folder= [i for i in os.listdir('yale') if i.isnumeric()]
    
    img_files = [os.path.join('yale', i, j) for i in img_folder for j in os.listdir('yale/' + i)]
        
    data = np.empty((504,0))
    
    for i in img_files:
        # read the pic and 
        pic_im = imread(i).flatten().astype('double')
        
        label.append(label_dic[i.split('/')[1]])
        
        tmp_doc_product = np.dot(R, np.array([pic_im]).T)
        
        data = np.append(data, tmp_doc_product, axis = 1)
        
    return data, label, R

########################################
 #       """ 
 #       initilization methods
 #       """
########################################
        
def initilize_dict(data):
    """
        Args:
            data: img data. shape is 504 * 316 
        return:
            Dict: normalized dict 
                [] size is 5, each element is 14 pictures. shape is 5 * 504 * 14
    """        
    pic_selection = np.random.choice(316, 5)  
    cluster_center_dic = [data[:,i:i+K] for i in pic_selection]
    
    cluster_center_dic = [dict_normalization(i) for i in cluster_center_dic]
    
    return cluster_center_dic

def cluster_data_with_dict(data, dict_of_data):
    """
        Args:
            data: img data
            dict_of_data: dict of data
        return:
            cluster labels
    """
    count_of_img = data.shape[1]
    T  = np.zeros((count_of_img, 1))

    for t in np.arange(count_of_img):
        temp = np.zeros((5,1))
        for i in range(C):
            omp = OMP(n_nonzero_coefs=S)
            omp.fit(dict_of_data[i], data[:,t])
            a = omp.coef_.T
            temp[i] = sum(abs(data[:,t] - dict_of_data[i].dot(a)))
        
        min_loc = temp.argmin(axis = 0)
        T[t] = min_loc
    
    return T
    
def get_class_assignment(data, clustered_labels, number_of_classes):
    """
        args:
            data: dataset to be cluster 505 * 316
            clustered_labels: cluster labels
            number_of_classes: number of classes
        return:
            class_assignment: the assignment of element
    """
    
    class_assignment = []
    
    clustered_labels = np.array(clustered_labels)
    
    for i in range(number_of_classes):
        class_assignment.append(data[:,np.where(clustered_labels == i)[0]])
    
    return class_assignment    

########################################
#        """ 
#        judge methods
#        
#        """
########################################
        
def score_of_accuracy(true_labels, clustered_labels):
    """
        args:
            true_labels: true labels of imgs.  shape is 316
            clustered_labels: labels generated from cluster method. shape the same with above
        return:
            score: the accuracy score
            cmat: not mandatory, take with _
    """
    n = len(true_labels)
    
    true_labels = np.array(true_labels)
    clustered_labels = np.array(clustered_labels).flatten()
    
    from scipy.sparse import coo_matrix
    
    cat = coo_matrix((np.ones((n,1)).flatten(), (np.array(np.arange(n)).T,true_labels)))
    cls = coo_matrix((np.ones((n,1)).flatten(), (np.array(np.arange(n)).T,clustered_labels)))

    cmat = cls.T.dot(cat)
    
    from scipy.optimize import linear_sum_assignment
    
    row_ind, col_ind = linear_sum_assignment(-cmat.toarray())
        
    score = float(100*(cmat[row_ind, col_ind].sum()/n))

    return score, cmat    


########################################
#""" 
#        learning methods
#        
#"""
########################################


def show_energy(cluster_dict, class_assignment, S, theta1, theta2):
    """
        todo
    """
    import numpy as np

    C = len(cluster_dict)
    n, K = cluster_dict[0].shape
    
    energy_r = 0
    energy_d1 = 0
    
    for i in range(C):
        
        omp = OMP(n_nonzero_coefs=4)
        omp.fit(cluster_dict[i], class_assignment[i])
        
        Acoff = omp.coef_
        
        err = class_assignment[i] - cluster_dict[i].dot(Acoff.T)
    
        energy_r = energy_r + (err ** 2).sum()
        
    c = np.zeros((n, C))
    
    for i in range(C):
        c[:,i] = cluster_dict[i].mean(axis=1) 
        
        err = cluster_dict[i] - np.kron(np.ones((1,K)).T, c[:,i]).T
        energy_d1 = energy_d1 + (err ** 2).sum()


    mean_of_c = c.mean(axis = 1)
    
    err = c - np.kron(np.ones((1, C)).T, mean_of_c).T

    energy_d2 = 14 * ((err ** 2).sum())
    
    energy_d = theta1*energy_d1 - theta2*energy_d2

    energy = energy_r + energy_d
    
    values = np.array([energy_r, energy_d1, energy_d2, energy_d])
        
    return energy, values



def fisher_discriminantor(class_assignment, params):
    """
        args:
            class_assignment:
            params:
                K: number of atmos in sub-dictionary
                S: sparse level
                theta1: trade-off
                theta2: trade-off
                numIteration: iteration gate
    """
    
    C = len(class_assignment)
    K = params.K
    S = params.S
    n, _ = class_assignment[1].shape
    theta1 = params.theta1
    theta2 = params.theta2
    numIteration = params.numIteration

    D = [dict_normalization(class_assignment[i][:,:14])  for i in range(C)]

    finished = aaa = 0
    
    ### douchbag
     
    P1 = np.zeros((C*K, C*K))
    temp = np.zeros((K, K))
    
    q = np.ones((K, 1)) / K
    
    for i in range(K):
        p = np.zeros((K, 1))
        p[i] = 1
        temp = temp + (p-q) * (p-q).T
        
    for i in range(C):
        P1[i*K : (i+1) * K , i * K : (i+1) * K] = temp
        
    P2 = np.zeros((C*K, C*K))
    s = np.ones((K*C, 1))/C
    
    for i in range(C):
        r = np.zeros((C*K, 1))
        r[i*K:(i+1)*K] = 1
        P2 = P2 + (r -s ).dot((r -s). T)
        
    T = theta1 * P1 - theta2 * P2
    
    for i in range(numIteration):
        Alpha = []
        for j in range(C):
            omp = OMP(S)
        
            omp.fit(D[j], class_assignment[j])
            
            Alpha.append(omp.coef_.T)
            
        Q = np.zeros((n, C*K))
        for jj in range(C):
            Q[:,jj*K:(jj+1)*K] = class_assignment[jj].dot(Alpha[jj].T)
         
            
        A = np.zeros((C*K, C*K))    
        for jjj in range(C):
            A[jjj*K:(jjj+1)*K, jjj*K:(jjj+1)*K] = Alpha[jjj].dot(Alpha[jjj].T)
        
        W = A + T
                
        if np.linalg.matrix_rank(W) == 0:            
            DO = np.linalg.solve(W.T, Q.T).T

        else:
            DO = np.dot(Q, np.linalg.pinv(W))
                
        DO[np.where(np.isnan(DO)==1)] = 0
        DO[np.where(np.isinf(DO)==1)] = 10**20
        
        DO = dict_normalization(DO)
        
        for j in range(C):
            D[j] = DO[:,j*K:(j+1)*K]
        
        bbb, _ = show_energy(D, class_assignment, S, theta1, theta2)
                
        if abs((bbb-aaa)/ bbb) < 0.002:
            if finished < 1:
                finished += 1
            else:
                pass
        else:
            finished = 0
            
        aaa = bbb
        
    return D
    
def assemble(data_dict, x, S):
    """
    """
    C = len(data_dict)
    temp = np.zeros((C,1))
    
    for i in range(C):
        omp = OMP(S)
        omp.fit(data_dict[i], x)
        a = omp.coef_
        temp[i] = abs(x - data_dict[i].dot(a)).sum(axis = 0)
    
    min_loc = temp.argmin(axis = 0)
    
    return min_loc
    

def loop_face_learning(data, data_dict, class_assignment, params, score):
    """
    """
    iteration_number= 80
    finished = 0
    locked = 0
    
    acc_arr = []
    
    energy = np.zeros((iteration_number, 1))
    values = np.zeros((iteration_number, 4))
    
    aaa, _ = show_energy(data_dict, class_assignment, params.S, params.theta1, params.theta2)

    for it in range(iteration_number):
        
        dic_cid = fisher_discriminantor(class_assignment, params)

        T = [assemble(dic_cid, data[:,i], 4) for i in range(316)]
        
        class_assignment = get_class_assignment(data, T, C)
        
        acc_arr.append(score_of_accuracy(img_label, T))
        
        energy[it],values[it,:]  = show_energy(data_dict, class_assignment, 4, params.theta1, params.theta2)
        
        bbb = energy[it]
        
        bbb_minus_aaa = abs((bbb-aaa)/bbb)
                
        if bbb_minus_aaa < 0.004:
            if finished < 2:
                finished += 1
            else:
                if params.theta2 < 36:
                    params.theta1 += 0.4
                    params.theta2 += 1
                    finished = 0
                else:
                    break
        elif bbb_minus_aaa < 0.04:
            finished = 0
            if locked < 3:
                locked += 1
            else:
                if params.theta2 < 36:
                    params.theta1 += 0.4
                    params.theta2 += 1
                    params.finished = 0
                else:
                    break
        else:
            locked = 0
            finished = 0 
        
        aaa = bbb
        
        print('iter %d \n  the precision is ' % it + str(acc_arr[it]) )
        score.append(acc_arr[it])
        
        
        
        


if __name__ == '__main__':
    img_data, img_label, R = read_pic_as_dict()
    img_data = dict_normalization(img_data)

    data_dict = initilize_dict(img_data)
    
    cluster_label = cluster_data_with_dict(img_data, data_dict)
    
    class_assignment = get_class_assignment(img_data, cluster_label, 5)
    
    score = []
    
    score.append(score_of_accuracy(img_label, cluster_label))
    
    loop_face_learning(img_data, data_dict,class_assignment, params, score)