#This example created a linear regression equation, then use a a simple
#TensorFloe (TF) NN to buildt the linear equation using training data
#NN parameters are then compared to the real parameters
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam


#Create the linear equation
N = 100
X = np.random.random(N) * 6 - 3
#y = mx+c, here m = .5 and c = -1 
y = 0.5 * X - 1 + np.random.randn(N) * 0.5 #np.random.randn(N) * 0.5 is Gaussian noise

# build model
i = Input(shape=(1,))
x = Dense(1)(i)

model = Model(i, x)

print(model.summary())

model.compile(
  loss='mse',
  # optimizer='adam',
  optimizer=Adam(learning_rate=0.1),
  metrics=['mae']
)

r = model.fit(
  X.reshape(-1, 1), y,
  epochs=200,
  batch_size=32,
)


# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.legend()

# Plot mae per iteration
plt.plot(r.history['mae'], label='mae')
plt.legend()


# Make predictions on new data
Xtest = np.linspace(-3, 3, 20).reshape(-1, 1)
ptest = model.predict(Xtest)

plt.scatter(X, y)
plt.plot(Xtest, ptest);

# Check the learned parameters
print(model.layers)
