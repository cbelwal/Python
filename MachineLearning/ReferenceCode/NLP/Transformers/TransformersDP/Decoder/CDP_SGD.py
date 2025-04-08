import torch
import numpy as np

# Ref: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
class CDP_SGD:
    def __init__(self, model, learning_rate=0.01, delta=0.5, eps = 1.0, C=1.0):
        self.model = model
        self.learning_rate = torch.tensor(learning_rate)
        self.σ = torch.tensor(self.getStandardDeviation(delta, eps))
        self.C = torch.tensor(C) # Clipping Threshold
        for param in model.parameters():
            param.accumulated_grads = []
    
    # This insures each step is (ε,δ)-dierentially private with respect to the lot 
    # Ref: https://arxiv.org/pdf/1607.00133.pdf
    def getStandardDeviation(self,δ, ε):
        σ = np.sqrt (2 * np.log(1.25/δ))
        σ /= ε
        return σ

    def stepForSingleSampleMatrixOp(self):
        allGrads = self.model.parameters().grad.detach().clone()   
        sanitizedGrads = self.getSanitizedGradients(allGrads)
        param = param - self.args.lr * sanitizedGrads
        self.model.parameters().grad = 0 # Reset for next iteration

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
        for param in self.model.parameters():
            # these gradients are already computed
            grads = param.grad.detach().clone()
            sanitizedGrads = self.getSanitizedGradients(grads)
            # This is what optimizer.step() does
            param = param - self.learning_rate * sanitizedGrads
            # Note: In the Medeium article, the noise is added after gradient is computed
            # which is at this point. But in Abadi et al. the noise is added before
            param.grad = 0 # Reset for next iteration
            count += 1
        #self.computePrivacyLoss(param.grad, param.data)
        print(count)

    def computePrivacyLoss(self, x, y):
        # Forward pass
        outputs = self.model(x)
        loss = torch.nn.functional.cross_entropy(outputs, y)
        return loss
        
            
            


    