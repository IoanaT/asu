import numpy as np
import random

data = np.load('AllSamples.npy')

def initial_point_idx2(id,k, N):
    random.seed((id+k))     
    return random.randint(0,N-1)

def initial_S2(id):
    print("Strategy 2: k and initial points")
    i = int(id)%150 
    random.seed(i+800)
    k1 = 4
    k2 = 6
    init_idx2 = initial_point_idx2(i, k1,data.shape[0])
    init_s1 = data[init_idx2,:]
    init_idx2 = initial_point_idx2(i, k2,data.shape[0])
    init_s2 = data[init_idx2,:]
    return k1, init_s1, k2, init_s2

def rest_of_k_and_Centroids(id):
    print("Strategy 1: rest of k 2,3,5,7,8,9,10 and their centroids")
    i = int(id)%150 
    random.seed(i+800)
    k3 = 2
    k4 = 3 
    k5 = 5
    k6 = 7
    k7 = 8
    k8 = 9
    k9 = 10
    
    init_idx = initial_point_idx2(i,k3,data.shape[0])
    init_s3 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k4,data.shape[0])
    init_s4 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k5,data.shape[0])
    init_s5 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k6,data.shape[0])
    init_s6 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k7,data.shape[0])
    init_s7 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k8,data.shape[0])
    init_s8 = data[init_idx,:]
    
    init_idx = initial_point_idx2(i,k9,data.shape[0])
    init_s9 = data[init_idx,:]
    
    return k3, init_s3, k4, init_s4, k5, init_s5, k6, init_s6, k7, init_s7, k8, init_s8, k9, init_s9