# This class shows how the embedding later in Pytorch work

import torch
import torch.nn as nn

#------------------
class EmbeddingsLayer(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        print(f"Input dim {input_dim}, embedding dim {embedding_dim}")
        self.embedding = nn.Embedding(input_dim, embedding_dim)

    def setWeightsToOne(self):
        nn.init.uniform(self.embedding.weight,1)

    def printWeights(self):
        print("Weights,",self.embedding.weight) 

    # src is the en_ids, or list of token ids in a sentence
    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.embedding(src) 
        return embedded
#------------------

data = [0,1,2,3,4]
tensorData = torch.tensor(data, device="cpu")

model = EmbeddingsLayer(10,30)
print("Weights before init:")
model.printWeights()
print("Weights after init:")
model.setWeightsToOne()
model.printWeights()

output = model(tensorData)
print("*** Model Output:",output)

