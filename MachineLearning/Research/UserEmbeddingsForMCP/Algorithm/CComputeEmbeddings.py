import torch
import torch.nn as nn
import torch.nn.functional as F

class CComputeEmbeddings():
    def train(model, x, y, epochs=50, lr=0.01):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            # For each tool per user
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print('Epoch:', epoch, 'Loss:', loss.item())
        print('Epoch:', epoch, 'Loss:', loss.item())

    # Function to compute cosine similarity between two vectors using numpy
    def compute_cosine_similarity(tensor1, tensor2):
        output = F.cosine_similarity(tensor1, tensor2, dim=0)
        return output

    def get_test_MAT_tau_u(totalNumberOfUsers, totalNumberOfTools):
        MAT_tau_u = torch.zeros(totalNumberOfUsers, totalNumberOfTools)
        # ------ Manually Assign Values
        MAT_tau_u[0][0] = 1 # user 0, tool 0
        MAT_tau_u[0][1] = 2 # user 0, tool 1
        MAT_tau_u[0][2] = 2 # user 0, tool 2
        MAT_tau_u[0][3] = 0 # user 0, tool 3
        MAT_tau_u[1][0] = 3 # user 1, tool 0
        MAT_tau_u[1][1] = 1 # user 1, tool 1
        MAT_tau_u[1][2] = 0 # user 1, tool 2
        MAT_tau_u[1][3] = 2 # user 1, tool 3
        # Canary #1: user 2 has same values as user 0
        MAT_tau_u[2][0] = 1 # user 2, tool 0
        MAT_tau_u[2][1] = 2 # user 2, tool 1
        MAT_tau_u[2][2] = 2 # user 2, tool 2
        MAT_tau_u[2][3] = 0 # user 2, tool 3
        # Canary #2: user 3 has close values as user 1
        MAT_tau_u[3][0] = 3 # user 1, tool 0
        MAT_tau_u[3][1] = 1 # user 1, tool 1
        MAT_tau_u[3][2] = 1 # user 1, tool 2 -> Only difference
        MAT_tau_u[3][3] = 2 # user 1, tool 3
        #------------------------
        return MAT_tau_u

if __name__== "__main__":
    # ---------- Data Generation
    embeddingDimensions = 3
    totalNumberOfTools = 4
    totalNumberOfUsers = 4
    # Create a Tensor of shape (1,4) with all elements set to 1
    MATx = torch.ones(embeddingDimensions,totalNumberOfTools)
    MAT_tau_u = get_test_MAT_tau_u(totalNumberOfUsers, totalNumberOfTools)


    print("Starting Model Training")
    # MATx shape: (4,2), MAT_tau_u shape: (3,2)
    
    
    tmpMATx = torch.ones(embeddingDimensions)
    MAT_E = torch.zeros(totalNumberOfUsers, embeddingDimensions)
    for i in range(0,totalNumberOfUsers):
        print(f"*** Execution for user {i}:")
        # Model will take the embedding dimenssions as input and return a single output containing value from  tool call.
        model = SingleLinear(embeddingDimensions, 1)
        
        # Get the ith row of MAT_tau_u
        # .view: reshape the tensor to be of shape (1, totalNumberOfTools)
        # the elements will be the same, just the shape will be different
        tmpMAT_tau_u = MAT_tau_u[i].view(1, totalNumberOfTools)
        
        # Input values of MATx should be in shape (noOfTools, noOfEmbeddings)
        # nn.linear expects the x input to be as a column vector
        # E.g. if there are 4 embeddings and 2 tools, then the input should be of shape (2,4)
        # and output will be of shape (2,1) 
        # nn.linear with treat the 2 tools as 2 independent samples which will me compared against
        # the value in MAT_tau_u
        # The 1st tool will be compared against 1st row in MAT_tau_u, 2nd on 2nd row
        # and so on. 
        # The Transpose operation is required to get them in the required shape
        train(model, MATx.T, tmpMAT_tau_u.T, epochs=15, lr=0.01)
        # Print all model parameters
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Copy values of param.data to row i of MAT_E
                MAT_E[i] = param.data  
                #print(name, param.data)
    
    print("Final Embedding Matrix MAT_E:")
    print(MAT_E)

    '''
    In PyTorch, the .item() method is used to extract the value 
    from a single-element tensor and convert it into a standard 
    Python number (e.g., int or float). 
    '''
    print("Cosine Similarity User 0 and User 2:", 
          compute_cosine_similarity(MAT_E[0], 
                                    MAT_E[2]).item())
    
    print("Cosine Similarity User 1 and User 3:", 
          compute_cosine_similarity(MAT_E[1], 
                                    MAT_E[3])
                                    .item())
    
    print("Cosine Similarity User 0 and User 1:", 
          compute_cosine_similarity(MAT_E[0], 
                                    MAT_E[1])
                                    .item())