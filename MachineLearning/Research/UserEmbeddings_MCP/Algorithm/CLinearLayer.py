import torch
import torch.nn as nn

class CLinearLayer(nn.Module):
    def __init__(self, embeddingDimensions=4, totalNumberOfTools=2):
        super().__init__()
        # Need to set seed for reproducibility, has to be defined 
        # by setting up layet as weights are assigned at that level
        torch.manual_seed(1) 
        # nn.Linear (input features, output features)
        self.linear = nn.Linear(embeddingDimensions,
                                totalNumberOfTools, bias=False)

    def forward(self, x):
        y = self.linear(x)
        return y