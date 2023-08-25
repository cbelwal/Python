import numpy as np
import matplotlib.pyplot as plt

#Specify config
D = 2   #Number of features, each dimension
K = 3   #Number of clusters = 3, with IDs 0,1, and 2
N = 300 #Number of samples

'''
#########################################################
Create the data and compute mean
#########################################################
'''
#Create data
#Create 3 distinct point, where the random values can be centered in
mu1 = np.array([0,0]) #array of 2 dimensions, with values 0,0
mu2 = np.array([5,5]) #array of 2 dimensions, with values 5,5
mu3 = np.array([0,5]) #array of 2 dimensions, with values 0,5

X = np.zeros((N,D)) #Matrix with N rows and D columns, all init to 0
#rndMatrix is a 100xD matrix of random numbers
rndMatrix = np.random.randn(100,D) #randn generates random numbers from a Gaussian distribution
#print(rndMatrix)
X[:100,:] = rndMatrix + mu1 #This addition will center the Matrix around mu1
X[100:200,:] = np.random.randn(100,D) + mu2 #Second set of number centered around mu2
X[200:,:] = np.random.randn(100,D) + mu3
#Create an array with 100 values each of 0,1,2 for each cluster
Y = np.array([0] * 100 + [1] * 100 + [2] * 100) 

#print(X)
#Visualize the data -> Data should show already in cluster
#plt.scatter(X[:,0],X[:,1],c=Y) # c is used to color
#plt.show()

#Avg 
means = np.zeros((K,D))
means[0] = np.mean(X[:100,:],axis=0)
means[1]  = np.mean(X[100:200,:],axis=0)
means[2] = np.mean(X[200:,:],axis=0)
#print(means)



#Plot data with mean
#plt.scatter(X[:,0],X[:,1],c=Y) # c is used to color
#plt.scatter(means[:,0],means[:,1],s=500,c='red',marker='*') # s is size of datapoint
#plt.show()

'''
#########################################################
Create new data and find cluster is belongs to
#########################################################
'''
newN = 30
newX = np.zeros((newN,D)) #30 new points
rndMatrix = np.random.randn(10,D) #randn generates random numbers from a Gaussian distribution
#print(rndMatrix)
newX[:10,:] = rndMatrix + mu1 #This addition will center the Matrix around mu1
newX[10:20,:] = np.random.randn(10,D) + mu2 #Second set of number centered around mu2
newX[20:,:] = np.random.randn(10,D) + mu3


Y = np.zeros(newN)

for i in range (newN):
    closest = (float('inf'),0)
    for j in range(K): #For each cluster
        #Dot product a.b =|a|.|b| cos θ,|a| and |b| is magnitude of vector
        #Here θ = 0, hence cos θ = 1,
        # if a = (a1,b1), |a| = sqrt(a1*a1 + b1*b1)
        # let a = newX[i] - means[j] = (x1 - x2,y1 - y2)
        # a.a = |a|.|a| cos 0 
        #     = sqrt((x1 - x2)*(x1 - x2),(y1 - y2)*(y1 - y2)) ...eq(1)
        # Distance formula between two point a(x1,y1) and y(x2,y2)
        #     = sqrt((x1 - x2)*(x1 - x2),(y1 - y2)*(y1 - y2))  
        # So eq(1) is the distance between two points and b
        euc = (newX[i] - means[j]).dot(newX[i] - means[j])
        if euc < closest[0]:
            closest = (euc,j)
            #print(j)
    Y[i] = closest[1] #Take the cluster

#print("Final Clusters",Y)
#plt.scatter(newX[:,0],newX[:,1],c=Y) # c is used to color
#plt.show()

'''
#########################################################
Create new random points and find the cluster center for it using Convergence
#########################################################
'''
#Create data
#Create 3 distinct point, where the random values can be centered in
mu1 = np.array([0,0]) #array of 2 dimensions, with values 0,0
mu2 = np.array([5,5]) #array of 2 dimensions, with values 5,5
mu3 = np.array([0,5]) #array of 2 dimensions, with values 0,5

X = np.zeros((N,D)) #Matrix with N rows and D columns, all init to 0

X[:100,:] = np.random.randn(100,D) + mu1 #This addition will center the Matrix around mu1
X[100:200,:] = np.random.randn(100,D) + mu2 #Second set of number centered around mu2
X[200:,:] = np.random.randn(100,D) + mu3

#Assign random centers
means = np.zeros((K,D))
rndIdx = np.random.randint(N,size=(K)) # 3 random points
means = X[rndIdx,:]

Y = np.zeros(N)
oldY = np.zeros(N)


#Plot data with mean
#---------------- Original Plot
#plt.scatter(X[:,0],X[:,1],c=Y) # c is used to color
#plt.scatter(means[:,0],means[:,1],s=500,c='red',marker='*') # s is size of datapoint
#plt.show()

costFunc = []
converged = False
idx = 0
while not converged:
    sum = 0
    converged = True
    print("Converged iteration:", idx)
    idx += 1
    for i in range (N):
        closest = (float('inf'),0)
        for j in range(K): #For each cluster
            euc = (X[i] - means[j]).dot(X[i] - means[j]) #This can also be **2
            if euc < closest[0]:
                closest = (euc,j)
            #print(j)
        oldY[i] = Y[i]
        Y[i] = closest[1] #Assign the cluster
        if(Y[i] != oldY[i]):
            converged = False

        sum += closest[0]

    costFunc.append(sum)

    #Compute new mean
    means[0,:] = X[Y==0].mean(axis=0)
    means[1,:] = X[Y==1].mean(axis=0)
    means[2,:] = X[Y==2].mean(axis=0)

print("Final Y:",Y)

#Plot data with mean
plt.scatter(X[:,0],X[:,1],c=Y) # c is used to color
plt.scatter(means[:,0],means[:,1],s=500,c='red',marker='*') # s is size of datapoint
plt.show()

#Plot cost function
plt.plot(costFunc)
plt.show()
        