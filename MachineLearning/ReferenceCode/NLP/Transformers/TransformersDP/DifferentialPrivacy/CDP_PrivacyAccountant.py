import torch

class CDP_PrivacyAccountant:
    """
    This class implements the privacy accountant for differential privacy.
    It is used to calculate the privacy loss of a differentially private algorithm.
    """
    def __init__(self, num_samples=1, batch_size = 1):     
        self.amortized_ratio = batch_size / num_samples
        self.num_samples = num_samples
        self._amortized_ε = 0.0
        self._amortized_δ = 0.0

    def computePrivacySpending(self, eps,delta):
        # Forward pass    
        tmp_ε = torch.log(1.0 + self.amortized_ratio * (torch.exp(eps -1.0)))
        self._amortized_ε += tmp_ε
        self._amortized_δ = delta * self.amortized_ratio
        
    
    def getPrivacySpent(self):
        # Forward pass
        return (torch.sqrt(self._amortized_ε), self._amortized_δ)

       