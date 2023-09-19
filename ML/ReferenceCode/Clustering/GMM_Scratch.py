import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Soft kMeans, also called as Fuzzy clustering
#{\displaystyle J(W,C)=\sum _{i=1}^{n}\sum _{j=1}^{c}w_{ij}^{m}\left\|\mathbf {x} _{i}-\mathbf {c} _{j}\right\|^{2}},

def getData(N=500,D=2):
    #Create 300 samples with 2 dimensions belonging to 3 classes
    sep = 5 #How separate are clusters. 
    mu1 = np.array([0,0]) # mean of cluster 1, array of 2 dimensions, with values 0,0
    mu2 = np.array([sep,sep]) # mean of cluster 2,array of 2 dimensions, with values 5,5
    mu3 = np.array([0,sep]) # mean of cluster 3,array of 2 dimensions, with values 0,5

    X = np.zeros((N,D)) #Matrix with N rows and D columns, all init to 0

    X[:200,:] = np.random.randn(200,D) + mu1 #This addition will center the Matrix around mu1
    X[200:400,:] = np.random.randn(200,D) + mu2 #Second set of number centered around mu2
    X[400:,:] = np.random.randn(100,D) + mu3

    return X #500 Random Points

def gmm(X,K=3,max_iter=20,smoothing=1e-2):
    N,D = X.shape #number of rows, columns (dimensions)
    R = np.empty((N,K))
    M = np.zeros((K,D)) #Mean
    C = np.zeros((K, D, D)) #Covariance matrix
    pi = np.ones(K)/K #init to 1/K

    #Set initial values
    # Set mean to 3 random points
    rndIdx = np.random.choice(N, K, replace=False) #Select K points randomly between 0 to N
    M = X[rndIdx]

    # init covariance to a Identity matrix
    for k in range(K):
        C[k] = np.eye(D) #Identify matrix of size D

    #------------ Start Convergence Loop for EM algorithm
    # These wil contain the output of applying the Gaussian function to each 
    # value of N in each cluster
    lls = []
    w_pdfs = np.zeros((N,K)) #PDF values
    for i in range(max_iter): #Run this constantly
        # ---------- Expectation Step, compute γ
        for k in range(K):
            for n in range(N):
                # This will output pdfValue (or Likelihood value) for X[n]
                # given mean M[k] and covariance matrix[k]
                # Note the equation for multi-variate Gaussian is based
                # on co-variance matrix and variance (which is used only in single variable Gaussian PDF)
                # is not used 
                pdfValue = multivariate_normal.pdf(X[n], M[k], C[k]) #single value
                w_pdfs[n,k] = pi[k] * pdfValue #This method is slow but verbrose

        # Slow but verbrose method
        # Note that this computation is derived from the Bayes rule
        for k in range(K):
            for n in range(N): #R is same as gamma(γ) in the equation
                R[n,k] = w_pdfs[n,k]/w_pdfs[n,:].sum()


        # -------- Maximization Step - Recalculate Params
        # Update mean and pi
        for k in range(K):
            Nk = R[:,k].sum()
            pi[k] = Nk / N
            M[k] = R[:,k].dot(X) / Nk #R is eq. to γ

            #Update covariance matrix
            delta = X - M[k] # Difference from mean
            # expand_dims expands the shape of an array, inserts a new axis ins the axis position
            #Rdelta = np.expand_dims(R[:,k], -1) * delta 
            #C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D)*smoothing # D x D
            C[k] = (np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in range(N)) / Nk) + np.eye(D)*smoothing

           
        #Check if coverged
        ll = np.log(w_pdfs.sum(axis=1)).sum()
        lls.append(ll)
        if i > 0:
            if np.abs(lls[i] - lls[i-1]) < 0.1:
                break

    #After convergence
    print("Solution has converged")
    plt.plot(lls)
    plt.title("Log-Likelihoods")
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)
    return R


def main():
    X  = getData()

    #Plot initial points
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    
    print("Starting GMM")
    gmm(X,3) #3 clusters


if __name__ == '__main__':
    main()