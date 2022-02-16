
# coding: utf-8

# In[1]:


from Precode import *
import numpy as np
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')


# In[2]:


global loss_fct_array 
loss_fct_array = []


# In[3]:


k1,i_point1,k2,i_point2 = initial_S1('0520') # please replace 0111 with your last four digit of your ID


# In[4]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[5]:


k3, i_point3, k4, i_point4, k5, i_point5, k6, i_point6, k7, i_point7, k8, i_point8, k9, i_point9 = rest_of_k_and_Centroids('0520')


# In[6]:


# print(k3)
# print(i_point3)
# print(k4)
# print(i_point4)
# print(k5)
# print(i_point5)
# print(k6)
# print(i_point6)
# print(k7)
# print(i_point7)
# print(k8)
# print(i_point8)
# print(k9)
# print(i_point9)


# In[7]:


def distance(center, p):
    dist = 0 
#     print(center.shape[0])
    dist += (center[0] - p[0]) ** 2 + (center[1] - p[1]) ** 2
    dist = np.sqrt(dist)
    return dist


# In[8]:


def loss_fct(center, p1):
    dist = 0 
#     print(center.shape[0])
    dist += (center[0] - p1[0]) ** 2 + (center[1] - p1[1]) ** 2
    return dist


# In[9]:


def update_centroids(k, clusters):
    centroids = []*k
    for count, cluster in enumerate(clusters):
        new_centroid = np.mean(cluster, axis = 0)
        centroids = np.append(centroids, new_centroid)
        centroids = np.reshape(centroids, (-1,2))
#         print("Count: ", count)
#         print("Cluster: ", cluster)
#         print("New centroids: ", centroids)
    return centroids


# In[10]:


def compare_centroids(oldCentroids, newCentroids):
    convCount = 0
    for i in range(len(oldCentroids)):
        diff = distance(oldCentroids[i], newCentroids[i])
        #print("Difference between old and new centroids:", diff)
        if diff == 0.0 : 
            convCount += 1
    
    #print("convergence count: ", convCount)
    if convCount == len(oldCentroids):
        return True
    
    return False


# In[11]:


def k_means(k, centroids, data):
    
    print("Number of clusters:", k)
    print("Centroids are: ", centroids)
    
    dataMemberships = [];
    dataDistances = [];
    clusters = []*k
    oldCentroids = centroids
    newCentroids = []*k
    countIterations = 0;
    global loss_fct_array

    while(True):
    
        dataMemberships = []
        dataDistances = []
        clusters = []
        
        
        for dataPoint in data:
            distances = []
            d = 0

            #for each data point calculate distances to centroids
            for i in range(k):
                d = distance(oldCentroids[i], dataPoint)
                distances = np.append(distances, d)
            #find index of minimum centroid
            min_centroid = np.argmin(distances)
            dataMemberships = np.append(dataMemberships, min_centroid)
            dataDistances = np.append(dataDistances, loss_fct(oldCentroids[min_centroid], dataPoint))

#         print("The loss function: ", dataDistances)
#         print("Which cluster does each point belong to: ", dataMemberships)
#         print(dataMemberships.shape[0])

        #construct clusters
        for i in range(k):
            cluster = np.array([])
            for j in range(dataMemberships.shape[0]):
                if (i == dataMemberships[j]):
                   # print("Data[j,:]:", data[j,:])
                    cluster = np.append(cluster, data[j,:])
                    cluster = np.reshape(cluster, (-1,2))
            clusters.append(cluster)   

        newCentroids = update_centroids(k, clusters)
        if compare_centroids(oldCentroids, newCentroids) == True : 
            loss_fct_array = np.append(loss_fct_array, np.sum(dataDistances, axis=0))
            print("Loss function: ", np.sum(dataDistances, axis=0))
            print("New centroids are: ", newCentroids)
            print("Iterations: ", countIterations)
            break
        oldCentroids = newCentroids
        countIterations += 1
        
#     total_loss = np.sum(dataDistances, axis = 0)        
#     print("Clusters are: ", clusters)


# In[16]:


def plot_cluster_size_vs_cost():
    clusterSize = [2,3,4,5,6,7,8,9,10]
    initPoints = [i_point3, i_point1, i_point4, i_point2, i_point5, i_point6, i_point7, i_point8, i_point9]
    for k in clusterSize:
        k_means(k, initPoints[k-2], data)
    plt.plot(clusterSize,loss_fct_array)
    plt.xlabel("Cluster Size")
    plt.ylabel("Loss Function")
    plt.title("K-Means Cluster Size vs Loss Fct.")
    plt.show()


# In[17]:


plot_cluster_size_vs_cost()


# In[18]:


k_means(k1, i_point1, data)


# In[19]:


k_means(k2, i_point2, data)

