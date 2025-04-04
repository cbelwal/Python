import torch
import numpy as np

# Ref: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
class CDP_SGD:
    def __init__(self, model, learning_rate=0.01, delta=0.5, eps = 1.0, C=1.0):
        self.model = model
        self.learning_rate = learning_rate
        self.σ = self.getStandardDeviation(delta, eps)
        self.C = C # Clipping Thershold
        for param in model.parameters():
            param.accumulated_grads = []
    
    # still steps insures each step is (ε,δ)-dierentially private with respect to the lot 
    # Ref: https://arxiv.org/pdf/1607.00133.pdf
    def getStandardDeviation(self,δ, ε):
        σ = np.sqrt (2 * np.log(1.25/δ))
        σ /= ε
        return σ


    def stepForSingleSample(self):
        for param in self.model.parameters():
            per_sample_grad = param.grad.detach().clone()
            # torch.nn.utils.clip_grad_norm_ defaults to using the L2 norm 
            # Lw Norm of vector is defined as sqrt(sum(x_i^2)) for i=1 to n
            torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=self.C)  # in-place
            # Add the noise
            # in papeer N(mean, variance) represenatation is used.
            param += torch.normal(mean=0, std=self.σ * self.C)
            # This is what optimizer.step() does
            param = param - self.args.lr * param.grad
            # Note: In the Medeium article, the noise is added after gradient is computed
            # which is at this point. But in Abadi et al. the noise is added before
            param.grad = 0 # Reset for next iteration
        
            
            


    