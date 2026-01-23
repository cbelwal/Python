"""
Embeddings are generated using a Polynomial reduction

Since this uses the PyTorch library, the sequence of steps will vary with the paper

"""
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))
sys.path.append(topRootPath)
#----------------------------------------------
from Algorithms.Helpers.IUserToolMatrix import IUserToolMatrix  

SCALING_FACTOR = 1.0

# Reduces values to the given dimension usine PCA
def __generate_pca_for_given_values__(values, reduce_to_dim:int=8):
    # Convert embeddings dict to a 2D array for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)
    model = PCA(n_components=reduce_to_dim)
    # Fit PCA to the scaled data and transform it
    pca_components = model.fit_transform(X_scaled)
    return pca_components

# Mainly for evaluation baseline via PCA
def Alg_Baseline_PCA_GenerateUserEmbeddings(embeddingDimensions:int=8,
                                       testData: IUserToolMatrix = None):
    MAT_u_tau = testData.get_MAT_u_tau()

    # Multiply all values by a scaling factor to improve training stability
    MAT_u_tau = MAT_u_tau * SCALING_FACTOR

    print("Starting Model Training")
    
    MAT_E = torch.zeros(testData.NumberOfUsers, embeddingDimensions)
    loss_for_each_user = torch.zeros(testData.NumberOfUsers)
    # Train the model for each user
    # For PCA we need to validate all samples together
    #for i in tqdm(range(0,testData.NumberOfUsers)):          
        
        # Get the ith row of MAT_tau_u
        # .view: reshape the tensor to be of shape (1, totalNumberOfTools)
        # the elements will be the same, just the shape will be different
    tmpMAT_u_tau = MAT_u_tau[i].view(1, testData.NumberOfTools)
        
        # Convert tmpMAT_u_tau to numpy for processing in CPolynomialFitReduction
    num_py_array = tmpMAT_u_tau.numpy()
        
    np_array = __generate_pca_for_given_values__(num_py_array,
                                                    reduce_to_dim=embeddingDimensions)
                                                      
        # Convert back to tensor
    MAT_E[i] = torch.tensor(np_array, dtype=torch.float32)
    loss_for_each_user[i] = torch.tensor(0., dtype=torch.float32)

                    
    return (MAT_E,loss_for_each_user)

   
