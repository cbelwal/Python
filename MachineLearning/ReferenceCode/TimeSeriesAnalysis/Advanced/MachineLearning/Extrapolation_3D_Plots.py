# We generate random data set and try different ML Models
# to extrapolate values. Extrapolcation is used to predict Y based on
# unobserved X and it is same as prediction

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plots

from sklearn.svm import SVR 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor # Simple NN

# Make the dataset
N = 1000
# Will generate X1 and X2, there are the two variables used for the prediction
X = np.random.random((N, 2)) * 6 - 3 # uniformly distributed between (-3, +3)
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1]) # y = cos(2.x1) + cos(3.x2)

# Plot values in 3D
# Plot it
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
# Make is a 3d projecton, else it will be plotted on a 2d grid 
ax = fig.add_subplot(111, projection='3d')
# Y values will be in Y axis while X1, and X2 wil be in 2 different X-axis
ax.scatter(X[:,0], X[:,1], Y)
plt.title("Original Equation: Raw data points")
plt.show()


#------------- *************** True function when plotted in -5x5 dimension
# This is useful when comparing the exptrapolated values for the ML models
# in 5,-5 dimenstion
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], Y)

# surface plot
line = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# Specify the true Y value as function of cos, as given for original values
Ytrue = np.cos(2*Xgrid[:,0]) + np.cos(3*Xgrid[:,1]) ## the true function
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Ytrue, linewidth=0.2, antialiased=True)
plt.title("Original Equation: Surface when expanded to -5,5")
plt.show()

# ------------ **************** SVM
# C Arguement controls the regularization
# Make sure to the use the Supprt Vector Regressor (SVR) class
# as we are doing a regression problem and not classification (SVC) class
# SVR does not show the boundary Hyperplane, but computes the actual values
model = SVR(C=100.) 
model.fit(X, Y)

# --------- Plot the data points on training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -3 to 3 which is where the training data lies
line = np.linspace(-3, 3, 50) 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("SVM: Surface after Prediction - Training Values (-3,3)")
plt.show()
# SVM does a good job in following the training lines 

# --------- Plot the data points on training and outside training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -5 to 5 which is where the training data lies
line = np.linspace(-5, 5, 50) # This is the only change from previous plot 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("SVM: Surface after Prediction - Outside Training Values (-5,5)")
plt.show()
# We can see the SVM model does not do a good job in extrapolation.
# We can see the SVM model does not do a good job in extrapolation.
# The plot is quite different from the true function plotted in -5,5 range
# as shown above


# ------------ **************** RANDOM FOREST
# C Arguement controls the regularization
# Make sure to use the RandomForestRegressor class as this is not a 
# classification problem
model = RandomForestRegressor()
model.fit(X, Y)

# --------- Plot the data points on training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -3 to 3 which is where the training data lies
line = np.linspace(-3, 3, 50) 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("Random Forest: Surface after Prediction - Training Values (-3,3)")
plt.show()
# Being more of a linear predictor
# one can see that the surface is not even 

# --------- Plot the data points on training and outside training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -5 to 5 which is where the training data lies
line = np.linspace(-5, 5, 50) # This is the only change from previous plot 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("Random Forest: Surface after Prediction - Outside Training Values (-5,5)")
plt.show()
# RF just uses last values to project data and does a poor job in extrapolation

# ------------ **************** MLPRegressor
# MLPRegressor is Multi-layer Perceptron, its a FF NN
model = MLPRegressor(hidden_layer_sizes=128, alpha=0., learning_rate_init=0.01)
model.fit(X, Y)

# --------- Plot the data points on training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -3 to 3 which is where the training data lies
line = np.linspace(-3, 3, 50) 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("MLP Regressor: Surface after Prediction - Training Values (-3,3)")
plt.show()
# Being more of a linear predictor
# one can see that the surface is not even 

# --------- Plot the data points on training and outside training values
fig = plt.figure(figsize=(10, 8))
# 111 means 1x1 grid, 1st subplot. So with 111, 
# there will be only 1 plot in the grid
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# Range of values is -5 to 5 which is where the training data lies
line = np.linspace(-5, 5, 50) # This is the only change from previous plot 
# Connect the lines as mesh
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
# The prediction from the model is regression, where it is trying to 
# predict the values (Yhat)
# based on different values of X (Xgrid)
Yhat = model.predict(Xgrid).flatten() # Prediction from model
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True);
plt.title("MLP Regressor: Surface after Prediction - Outside Training Values (-5,5)")
plt.show()
# MLP seems to give a more linear output
