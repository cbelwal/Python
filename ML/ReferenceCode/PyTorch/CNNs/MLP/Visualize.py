import matplotlib.pyplot as plt
import numpy as np
import torch

class Visualize:
#%matplotlib inline
    def viewTrainData(self,train_loader):    
        # obtain one batch of training images
        dataiter = iter(train_loader)
        images, labels = next(dataiter) #dataiter.next()
        images = images.numpy()

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            # print out the correct label for each image
            # .item() gets the value contained in a Tensor
            ax.set_title(str(labels[idx].item()))
        plt.show()

    def viewSingleImage(self,idx,train_loader):
        dataiter = iter(train_loader)
        images, labels = next(dataiter) #dataiter.next()
        images = images.numpy()
        img = np.squeeze(images[idx])

        fig = plt.figure(figsize = (12,12)) 
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y],2) if img[x][y] !=0 else 0
                ax.annotate(str(val), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y]<thresh else 'black')
        plt.show()


    def viewTestData(self,test_loader,model):    
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()

        # get sample outputs
        output = model(images)
        # convert output probabilities to predicted class
        _, preds = torch.max(output, 1)
        # prep images for display
        images = images.numpy()

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                        color=("green" if preds[idx]==labels[idx] else "red"))
        plt.show()