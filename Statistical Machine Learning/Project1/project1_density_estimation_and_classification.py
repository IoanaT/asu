
# coding: utf-8

# In[3]:


import numpy as np
import scipy.io
import math
import geneNewData

def main():
    myID='0520'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')

    features_0 = extract_features(train0)
    features_1 = extract_features(train1)
    
    global mean_feat1_0, var_feat1_0, mean_feat2_0, var_feat2_0, prior_prob_digit0, prior_prob_digit1
    global mean_feat1_1, var_feat1_1, mean_feat2_1, var_feat2_1
    
    # Feature 1 for digit 0 is the first column of the features_0 2D array
    mean_feat1_0, var_feat1_0 = calculate_mean_var(features_0[:,0])
    print('Mean feature1 digit0: ', mean_feat1_0)
    print('Variance feature1 digit0: ', var_feat1_0)
    
    # Feature 2 for digit 0 is the 2nd column of the features_0 2D array
    mean_feat2_0, var_feat2_0 = calculate_mean_var(features_0[:,1])
    print('Mean feature2 digit0: ', mean_feat2_0)
    print('Variance feature2 digit0: ', var_feat2_0)
    
    # Feature 1 for digit 0 is the first column of the features_0 2D array
    mean_feat1_1, var_feat1_1 = calculate_mean_var(features_1[:,0])
    print('Mean feature1 digit1: ', mean_feat1_1)
    print('Variance feature1 digit1: ', var_feat1_1)
    
    # Feature 2 for digit 0 is the 2nd column of the features_0 2D array
    mean_feat2_1, var_feat2_1 = calculate_mean_var(features_1[:,1])
    print('Mean feature2 digit1: ', mean_feat2_1)
    print('Variance feature2 digit1: ', var_feat2_1)
    
    prior_prob_digit0 = 0.5
    prior_prob_digit1 = 0.5
    
    global test0_features, test1_features
    
    test0_features = extract_features(test0)
    test1_features = extract_features(test1)
    
    print('test 1 features length:', len(test1_features))
    
    global y_hat_0, y_hat_1, y_hat_test0, y_hat_test1
    
    # calculate predicted label y_hat_test0 for testset with label 0 = test0 dataset
    
    y_hat_test0 = calculate_predicted_label(test0_features)
    print('Predicted labels for test set 0: ', y_hat_test0)
    print(len(y_hat_test0))
    
    # calculate predicted label y_hat_test1 for testset with label 1 = test1 dataset
    
    y_hat_test1 = calculate_predicted_label(test1_features)
    print('Predicted labels for test set 1: ', y_hat_test1)
    print(len(y_hat_test1))
                
    correct = 0
    for label in y_hat_test0: 
        if np.equal(label, 0):
            correct = correct + 1
        
    accuracy_0 = correct/len(test0) * 100
    
    print('Accuracy for digit 0 prediction: ', accuracy_0)
    
    correct = 0
    for label in y_hat_test1: 
        if np.equal(label, 1):
            correct = correct + 1
    accuracy_1 = correct/len(test1) * 100
    
    print('Accuracy for digit 1 prediction: ', accuracy_1)
    
    pass

# Calculating the Mean and Standard Deviation of each image belonging to a trainset 
# and appending the two features as an array containing 2D data points together
def extract_features(train_set):
    features = np.empty((0,2), int)
    for image in train_set:
        mean_img = np.mean(image)
        std_img = np.std(image)
        features = np.append(features, np.array([[mean_img, std_img]]), axis = 0)
    return features

def calculate_mean_var(features):
    mean = np.mean(features)
    var = np.var(features)
    return mean, var

# P(y) = (probability of mean/feature1)*(probability of std/feature2)*prior(respective class)
# x1 = feature1
# x2 = feature2
def classifier_digit0(x1, x2):
    prob_feature1 = p_y_given_x(x1, mean_feat1_0, var_feat1_0)
    prob_feature2 = p_y_given_x(x2, mean_feat2_0, var_feat2_0)
    return prior_prob_digit0 * prob_feature1 * prob_feature2


# P(y) = (probability of mean/feature1)*(probability of std/feature2)*prior(respective class)
# x1 = feature1
# x2 = feature2
def classifier_digit1(x1, x2):
    prob_feature1 = p_y_given_x(x1, mean_feat1_1, var_feat1_1)
    prob_feature2 = p_y_given_x(x2, mean_feat2_1, var_feat2_1)
    return prior_prob_digit1 * prob_feature1 * prob_feature2
    

def p_y_given_x(x, mean, variance):
    prob = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(x - mean) ** 2 / (2 * variance))
    return prob


# calculate posterior probabilities y_hat_0, y_hat_1 for each data point in the testset
# final PREDICTED LABEL y_hat of data point will be the higher probability among the two (y_hat_0, y_hat_1)
# both classifiers for digit 0 & digit 1 will be used
def calculate_predicted_label(testset):
    y_hat = np.empty((0,1), int)
    for data in testset: 
            y_hat_0 = classifier_digit0(data[0],data[1])
            y_hat_1 = classifier_digit1(data[0], data[1])
            if (y_hat_0 > y_hat_1):
                y_hat = np.append(y_hat, 0)
            else:
                y_hat = np.append(y_hat, 1)
    return y_hat

if __name__ == '__main__':
    main()
    

