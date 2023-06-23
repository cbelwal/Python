import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

#%matplotlib inline

# Read in the image
folder = './CNNs/CNN/data'
image = mpimg.imread(folder + '/curved_lane.jpg')


plt.imshow(image)
plt.show()

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()
# Create a custom kernel

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([[ -1, 0, 1], [ -2, 0, 2], [ -1, 0, 1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, sobel_y)

plt.imshow(filtered_image, cmap='gray')
plt.show()