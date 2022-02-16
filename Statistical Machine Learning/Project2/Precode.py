import numpy as np
import random

data = np.load('AllSamples.npy')

def initial_point_idx(id, k,N):
	return np.random.RandomState(seed=(id+k)).permutation(N)[:k]

def init_point(data, idx):
    return data[idx,:]

def initial_S1(id):
    print("Strategy 1: k and initial points")
    i = int(id)%150 
    random.seed(i+500)
    k1 = 3
    k2 = 5 
    init_idx = initial_point_idx(i,k1,data.shape[0])
    init_s1 = init_point(data, init_idx)
    init_idx = initial_point_idx(i,k2,data.shape[0])
    init_s2 = init_point(data, init_idx)
    return k1, init_s1, k2, init_s2

def rest_of_k_and_Centroids(id):
    print("Strategy 1: rest of k 2,4,6,7,8,9,10 and their centroids")
    i = int(id)%150 
    random.seed(i+500)
    k3 = 2
    k4 = 4 
    k5 = 6
    k6 = 7
    k7 = 8
    k8 = 9
    k9 = 10
    
    init_idx = initial_point_idx(i,k3,data.shape[0])
    init_s3 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k4,data.shape[0])
    init_s4 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k5,data.shape[0])
    init_s5 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k6,data.shape[0])
    init_s6 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k7,data.shape[0])
    init_s7 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k8,data.shape[0])
    init_s8 = init_point(data, init_idx)
    
    init_idx = initial_point_idx(i,k9,data.shape[0])
    init_s9 = init_point(data, init_idx)
    
    return k3, init_s3, k4, init_s4, k5, init_s5, k6, init_s6, k7, init_s7, k8, init_s8, k9, init_s9