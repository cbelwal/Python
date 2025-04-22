import torch
import numpy as np

# Ref: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
class CDP_SGD:
    def __init__(self, 
                 model,
                 totalSamples, 
                 learning_rate=0.01, 
                 delta=1e-7, 
                 eps = 1.0,
                 max_eps=64.0,
                 max_delta=1e-5,
                 C=1.0):
        self.model = model
        self.totalSamples = totalSamples
        self.learning_rate = torch.tensor(learning_rate)
        self.σ = torch.tensor(self.getStandardDeviation(delta, eps))
        self.max_eps = torch.tensor(max_eps)
        self.max_delta = torch.tensor(max_delta)
        self.C = torch.tensor(C) # Clipping Threshold
        
    
    # This insures each step is (ε,δ)-dierentially private with respect to the lot 
    # Ref: https://arxiv.org/pdf/1607.00133.pdf
    def getStandardDeviation(self,δ, ε):
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        σ = np.sqrt (2.0 * np.log(1.25/δ))
        σ /= ε
        return σ

    def getPrivacySpent(self):
       pass

    def getSanitizedGradients(self, grads):
        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(grads, max_norm=self.C)
        # Add noise to the gradients
        # in paper N(mean, variance) represenatation is used.
        # via Strong Composition Theorem
        # Each step is (ε,δ)-differentially private with respect to the lot. 
        # q = L/N, which is sampling ratio. L = 1 in our case
        # Each step is (O(q.ε),q.δ)-differentially private with respect to the whole database
        normalDist = torch.normal(mean=torch.tensor(0.0), std=self.σ * self.C) 
        grads += normalDist
        return grads

    def singleStep(self):
        count = 0        
        for param in self.model.parameters(): # Loop executed ~37 times
            # these gradients are already computed
            grads = param.grad.detach().clone()
            tensorInit = torch.zeros(grads.shape)
            sanitizedGrads = self.getSanitizedGradients(grads)
            # This is what optimizer.step() does
            param = param - self.learning_rate * sanitizedGrads
            # Note: In the Medeium article, the noise is added after gradient is computed
            # which is at this point. But in Abadi et al. the noise is added before
            # This is equivalent to param.zero_grad() in Optimizer
            param.grad = tensorInit # Reset for next iteration
            count += 1
        #self.computePrivacyLoss(param.grad, param.data)
        #print(count)

    def computePrivacyLoss(self, x, y):
        # Forward pass
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        return loss
    
    def usedUpPrivacyBudget(self, x, y):
        # Forward pass
        return False
        
            
            


    