
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')


# In[2]:


global loss_fct_array
loss_fct_array=[]


# In[3]:


k1,i_point1,k2,i_point2 = initial_S2('0520') # please replace 0111 with your last four digit of your ID


# In[4]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[5]:


# print(data)
print(data.shape)


# In[6]:


k3, i_point3, k4, i_point4, k5, i_point5, k6, i_point6, k7, i_point7, k8, i_point8, k9, i_point9 = rest_of_k_and_Centroids('0520')


# In[7]:


def distance(center, p):
    dist = 0 
    #print(center.shape[0])
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
#             print("Count: ", count)
#             print("Cluster: ", cluster)
#             print("New centroids: ", centroids)
            centroids = np.reshape(centroids, (-1,2))
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


def avg_dist_centroids(point, centroids):
    dist = []
    for c in centroids:
        #print("Centroid: ", c)
        dist = np.append(dist, distance(c, point))
    return np.mean(dist, axis=0)


# In[12]:


def init_centroids_strategy2(k, firstCentroid, data):
    centroids = np.array([])
    centroids = np.append(centroids, firstCentroid)
    print("Centroids at begin initialization: ", centroids)
    centroids = np.reshape(centroids, (-1,2))
    #for each centroid iterate all samples to find max distances to prev. k-1 centroids
    
    for i in range(k-1):
        dist = []
        for dataPoint in data:
            #avoid duplicate centroids
            if dataPoint in centroids:
                dist = np.append(dist, -1)
            else:
                dist = np.append(dist, avg_dist_centroids(dataPoint, centroids))
        max_dist_to_centroids_index = np.argmax(dist)
        #print("Index of max avg distance: ", max_dist_to_centroids_index)
        next_centroid = data[max_dist_to_centroids_index]
        #print("next_centroid: ", next_centroid)
        centroids = np.reshape(centroids, (-1,2))
        centroids = np.append(centroids, next_centroid)
        centroids = np.reshape(centroids, (-1,2))
        #print("Centroids after adding next centroid: ", centroids)
    return centroids       


# In[13]:


def k_means(k, centroids, data):
    
    print("Number of clusters:", k)
    #print("Centroids are: ", centroids)
    
    dataMemberships = []
    dataDistances = []
    clusters = []*k
    oldCentroids = centroids
    newCentroids = []*k
    countIterations = 0
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


# In[14]:


def plot_cluster_size_vs_cost():
    clusterSize = [2,3,4,5,6,7,8,9,10]
    initPoints = [i_point3, i_point4, i_point1, i_point5, i_point2, i_point6, i_point7, i_point8, i_point9]
    for k in clusterSize:
        centroids = init_centroids_strategy2(k, initPoints[k-2], data)
        k_means(k, centroids, data)
    plt.plot(clusterSize,loss_fct_array)
    plt.xlabel("Cluster Size")
    plt.ylabel("Loss Function")
    plt.title("K-Means Cluster Size vs Loss Fct.")
    plt.show()
        


# In[15]:


plot_cluster_size_vs_cost()


# In[18]:


centroids = init_centroids_strategy2(k1, i_point1, data)
print("Initial centroids: ", centroids)
k_means(k1, centroids, data)


# In[19]:


centroids = init_centroids_strategy2(k2, i_point2, data)
print("Initial centroids: ", centroids)
k_means(k2, centroids, data)

