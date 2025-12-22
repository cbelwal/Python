"""
Embeddings are generated using a single layer NN

Since this uses the PyTorch library, the sequence of steps will vary with the paper

"""
import torch
from tqdm import tqdm

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)
#----------------------------------------------
from Algorithms.Helpers.CSingleLayer import CSingleLayer
from Algorithms.Helpers.CModelTraining import CModelTraining
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix  

MAX_EPOCHS = 1000
LEARNING_RATE = 0.01
SCALING_FACTOR = 1.0
MIN_TARGET_LOSS = 1e-4


def Algorithm_2_GenerateUserEmbeddings(embeddingDimensions:int=8,
                                       testData: IUserToolMatrix = None):
    MAT_u_tau = testData.get_MAT_u_tau()

    # Multiply all values by a scaling factor to improve training stability
    
    MAT_u_tau = MAT_u_tau * SCALING_FACTOR

    print("Starting Model Training")
    # MATx shape: (4,2), MAT_tau_u shape: (3,2)
    MATx = torch.ones(embeddingDimensions, testData.NumberOfTools)
    #tmpMATx = torch.ones(embeddingDimensions)
    
    MAT_E = torch.zeros(testData.NumberOfUsers, embeddingDimensions)
    loss_for_each_user = torch.zeros(testData.NumberOfUsers)
    # Train the model for each user
    for i in tqdm(range(0,testData.NumberOfUsers)):
        # Model will take the embedding dimenssions as input and return a single output containing value from  tool call.
        model = CSingleLayer(embeddingDimensions)
        
        # Get the ith row of MAT_tau_u
        # .view: reshape the tensor to be of shape (1, totalNumberOfTools)
        # the elements will be the same, just the shape will be different
        tmpMAT_u_tau = MAT_u_tau[i].view(1, testData.NumberOfTools)
        
        # Input values of MATx should be in shape (noOfTools, noOfEmbeddings)
        # nn.linear expects the x input to be as a column vector
        # E.g. if there are 4 embeddings and 2 tools, then the input should be of shape (2,4)
        # and output will be of shape (2,1) 
        # nn.linear with treat the 2 tools as 2 independent samples which will me compared against
        # the value in MAT_tau_u
        # The 1st tool will be compared against 1st row in MAT_tau_u, 2nd on 2nd row
        # and so on. 
        # The Transpose operation is required to get them in the required shape
        #print("****\n",tmpMAT_u_tau)

        (model,loss) = CModelTraining.train(model, 
                                            MATx.T, 
                                            tmpMAT_u_tau.T, 
                                            max_epochs=MAX_EPOCHS,
                                            min_target_loss = MIN_TARGET_LOSS, 
                                            lr=LEARNING_RATE)
        loss_for_each_user[i] = loss.detach().clone() #torch.tensor(loss, dtype=torch.float32)

        # Print all model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Only store weights, not bias
                if name == 'linear.weight':
                    # Copy values of param.data to row i of MAT_E
                    MAT_E[i] = param.data                      
                    #print(name, param.data,loss)
    return (MAT_E,loss_for_each_user)

   
