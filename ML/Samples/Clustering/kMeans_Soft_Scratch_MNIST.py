#This uses the MNIST Dataset and also applies the 
#Purity and Davies-Bouldon Index to measure performance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kMeans_Soft_Scratch import compute_kmeans

#Return data from the MNIST data set
def getData(maxRows = None):
    #Create 300 samples with 2 dimensions belonging to 3 classes
    
    #Dataset from: https://www.kaggle.com/c/digit-recognizer
    #This dataset contains the flattened values of the 28x28 images in MNIST
    #Total samples 42000 and 28x28 = 784,
    # Matrix size is 42000 x 784  
    input_file= 'c:/users/chaitanya belwal/.datasets/clustering/MNIST/train.csv'
    df = pd.read_csv(input_file)
    data = df.values #Transfer to data for reshuffle purposes
    np.random.shuffle(data)
    # X = df.values[:,1:] #first column has target values
    # division by 255 is very important, else values becone nan as exponent 
    # values are close to 0 when e^(-beta * X) is taken 
    X = data[:,1:] / 255.0 
    Y = df.values[:,0]

    if maxRows is not None:
        X = X[:maxRows]
        Y = Y[:maxRows]

    return X,Y

#max purity = 1 : Higher the better 
#find points in each cluster and what is their true label
#The true label is the class to which most points belong to
# We need to know the true label, so this is supervised approach 
# Purity is an 'external validation' method
def purity(Y, R):
    # maximum purity is 1, higher is better
    N, K = R.shape
    p = 0
    for k in range(K): #For each class
        best_target = -1 # we don't strictly need to store this
        max_intersection = 0
        for j in range(K): #Go thorough all possible classes
            #This returns the sum of probabilities for all Y which belong to class j
            #Y==j is the index, return each sample from R
            #Each row x in R, it has feature data corresponding to correct label in row x of Y 
            #So we are computing the sum of R values, where the label corresponds to the actual label

            intersection = R[Y==j, k].sum() #R has the probabilities, Y is target label
            if intersection > max_intersection:
                max_intersection = intersection #find max_intersection for each k
                best_target = j
        p += max_intersection #For each k
    return p / N #This is the formula for purity

#Davies-Bouldin Index: Lower the better
#Measure the distance between clusters and cohesiveness between each cluster
#The idea is that the distance between clusters should be as high as possible
#and the distance of points in a single cluster should be as low as possible
#Kind of like: Low Coupling and High Cohesiveness used in S/W Engg.
#This is an 'internal validation' method
def DBI(X, M, R):
    # ratio between sum of std deviations between 2 clusters / distance between cluster means
    # lower is better
    N, D = X.shape
    K, _ = M.shape

    # get sigmas first
    sigma = np.zeros(K)
    for k in range(K):
        diffs = X - M[k] # should be NxD
        squared_distances = (diffs * diffs).sum(axis=1) # now just N
        weighted_squared_distances = R[:,k]*squared_distances
        sigma[k] = np.sqrt( weighted_squared_distances.sum() / R[:,k].sum() )

    # calculate Davies-Bouldin Index
    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                numerator = sigma[k] + sigma[j] #Sum of Std Devs in two clusters: lower the better
                #.norm() returns norm of a matrix. 
                # Norm is highest value in matrix
                # and is unrelated to the matrix dimensions
                # It is essentially converting the matrix to a scalar value
                denominator = np.linalg.norm(M[k] - M[j]) #Distance between means: Higher the better
                ratio = numerator / denominator
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K
            
def main():
    X,Y  = getData(1000) #Get 1000 datapoints
    print(X[2])

    K = len(set(Y)) #Set(Y) will give only the unique elements in Y
    print("Set:",set(Y) )
    M,R = compute_kmeans(X,K,show_plots=True) 

    print("DBI:", DBI(X, M, R))
    print("Purity:", purity(Y, R)) #Y is the target label

    # plot the mean images
    # they should look like digits
    for k in range(len(M)):
        im = M[k].reshape(28, 28)
        plt.imshow(im, cmap='gray') #Image show
        plt.show()

if __name__ == '__main__':
    main()

