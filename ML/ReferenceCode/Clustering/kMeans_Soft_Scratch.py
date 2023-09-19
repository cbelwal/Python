import numpy as np
import matplotlib.pyplot as plt

#Soft kMeans, also called as Fuzzy clustering
#{\displaystyle J(W,C)=\sum _{i=1}^{n}\sum _{j=1}^{c}w_{ij}^{m}\left\|\mathbf {x} _{i}-\mathbf {c} _{j}\right\|^{2}},

def getData(N=300,D=2):
    #Create 300 samples with 2 dimensions belonging to 3 classes
    
    mu1 = np.array([0,0]) #array of 2 dimensions, with values 0,0
    mu2 = np.array([5,5]) #array of 2 dimensions, with values 5,5
    mu3 = np.array([0,5]) #array of 2 dimensions, with values 0,5

    X = np.zeros((N,D)) #Matrix with N rows and D columns, all init to 0

    X[:100,:] = np.random.randn(100,D) + mu1 #This addition will center the Matrix around mu1
    X[100:200,:] = np.random.randn(100,D) + mu2 #Second set of number centered around mu2
    X[200:,:] = np.random.randn(100,D) + mu3

    return X

# Compute dot product of difference between u and v
# Note that once we compute diff. then dot product with itself will be
# simple multiplication of elements.
# We are not using mul (*) operator here as the operation is between vectors
# and not scalars
def dotDiff(u, v):
    diff = u - v
    return diff.dot(diff)

#Main cost function
#Note in documention R is also shown as 'w' corresponding to weights
#R is based on exp. function, this is different than what is shown in 
#some documents like of Wikipedia
def cost(X,R,M):
    cost = 0
    for k in range(len(M)):
        # method 1
        # for n in range(len(X)):
        #     cost += R[n,k]*d(M[k], X[n])

        # method 2
        diff = X - M[k] #M: mean / cluster center for each cluster
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost

def compute_kmeans(X, K, max_iter=20, beta=3.0, show_plots=False):
    N,D = X.shape #number of rows, columns (dimensions)
    exponents = np.empty((N,K)) #exponents for each classification

    # initialize M to random
    initial_centers = np.random.choice(N, K, replace=False) #Select K points randomly between 0 to N
    M = X[initial_centers] #Assign the K selected points

    costs = []
    

    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                # np.exp computes e ^ x, e=2.718, 
                # since its multiplied by -beta, the higher the dotdiff, the lower the exponent value will be
                # Hence it is important that X values are not very high as it can lead to 
                # very very low values of exponents which makes them ~0. This will results in division by 0
                # when values are divided by exponent sums like in computing R
                exponents[n,k] = np.exp(-beta*dotDiff(M[k], X[n])) #distance from mean
        # R is a fraction between 0 and 1
        # R determines the probability of belonging to each cluster
        # R is the main influencer when the mean is computed
        # R somewhat eq. to softmax (due to exp.)
        R = exponents / exponents.sum(axis=1, keepdims=True)

        # step 2: recalculate means
        # decent vectorization
        # This is weighted arithmetic mean
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()
        # oldM = M
        #print("Updated Means",M)
        # full vectorization
        #M = R.T.dot(X) / R.sum(axis=0, keepdims=True).T
        # print("diff M:", np.abs(M - oldM).sum())

        c = cost(X,R,M)
        costs.append(c) #Store hitory of cost function values

        if i>0: #Not the initial iteration
            #is delta between two costs functions is less than threshold
            #break as solution has converged
            #Otherwise loop will break out when required number of iterations are complete
            if np.abs(costs[-1] - costs[-2]) < 1e-5:
                break

    #Plot values
    if show_plots:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((K, 3)) #3 rows with 3 columns of random values
        colors = R.dot(random_colors) #colors will be NxK matrix
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()

    print("Final cost", costs[-1]) #Last value of costs
    return M, R

            
def main():
    X  = getData()
    compute_kmeans(X,K=3,show_plots=True)


if __name__ == '__main__':
    main()

