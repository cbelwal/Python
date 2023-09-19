import matplotlib.pyplot as plt
import numpy as np

class Visualize:
    def single_image(self,img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # c
        #plt.show()
    
    def get_imageByIdx(self,imageLoader,idx):
        dataiter = iter(imageLoader)
        images, labels = next(dataiter)
        images = images.numpy() # convert images to numpy for display
        return images[idx]

    def batch(self,imageLoader,classes):
        dataiter = iter(imageLoader)
        images, labels = next(dataiter)
        images = images.numpy() # convert images to numpy for display
        images.shape # (number of examples: 20, number of channels: 3, pixel sizes: 32x32)
    
        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display 20 images
        for idx in np.arange(20):
            ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
            self.single_image(images[idx])
            ax.set_title(classes[labels[idx]])
        plt.show()
    

    def single_image_details(self,img):
        rgb_img = np.squeeze(img)
        channels = ['red channel', 'green channel', 'blue channel']

        fig = plt.figure(figsize = (36, 36)) 
        for idx in np.arange(rgb_img.shape[0]):
            ax = fig.add_subplot(1, 3, idx + 1)
            img = rgb_img[idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(channels[idx])
            width, height = img.shape
            thresh = img.max()/2.5
            for x in range(width):
                for y in range(height):
                    val = round(img[x][y],2) if img[x][y] !=0 else 0
                    ax.annotate(str(val), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center', size=8,
                            color='white' if img[x][y]<thresh else 'black')
        plt.show()
