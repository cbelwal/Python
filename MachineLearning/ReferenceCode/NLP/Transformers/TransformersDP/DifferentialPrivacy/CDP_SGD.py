import torch
from DifferentialPrivacy.CDP_PrivacyAccountant import CDP_PrivacyAccountant

# Use torch tensors for all operations to prevent confusion between numpy and torch
# Ref: https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3
class CDP_SGD:
    def __init__(self, 
                 model,
                 totalSamples=1,
                 batch_size=1,
                 learningRate=0.01, 
                 delta=1e-7, 
                 eps = 1.0,
                 max_eps=64.0,
                 max_delta=1e-5,
                 C=1.0):
        self.model = model
        self.ε = torch.tensor(eps)
        self.δ = torch.tensor(delta)
        self.amortized_ratio = batch_size / totalSamples
        self.σ = self.getStandardDeviation() # returns a tensor
        self.learning_rate = torch.tensor(learningRate)
        self.max_eps = torch.tensor(max_eps)
        self.max_delta = torch.tensor(max_delta)
        self.C = torch.tensor(C) # Clipping Threshold
        self.privacyAccountant = CDP_PrivacyAccountant(num_samples=totalSamples, batch_size=1)
        
    
    # This insures each step is (ε,δ)-dierentially private with respect to the lot 
    # Ref: https://arxiv.org/pdf/1607.00133.pdf
    #
    # Will Laplace distribution ε is sampled from mean = 0 and SD = sqrt(2)/ε
    # σ = sqrt(2 * log(1.25/δ)) / ε  
    def getStandardDeviation(self):
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        #amortized_δ = self.amortized_ratio * self.δ
        σ = torch.sqrt (2.0 * torch.log(1.25/self.δ))
        σ /= self.ε
        return σ


    def getSanitizedGradients(self, grads):
        # Clip the gradients
        #torch.nn.utils.clip_grad_norm_(grads, max_norm=self.C)
        # Add noise to the gradients
        # in paper N(mean, variance) represenatation is used.
        # via Strong Composition Theorem
        # Each step is (ε,δ)-differentially private with respect to the lot. 
        # q = L/N, which is sampling ratio. L = 1 in our case
        # Each step is (O(q.ε),q.δ)-differentially private with respect to the whole database
        normalDist = torch.normal(mean=torch.tensor(0.0), std=self.σ * self.C) 
        grads += normalDist
        self.privacyAccountant.computePrivacySpending(self.ε, self.δ)
        return grads


    # For purpose of clarity, we are using a separate function for zero grad
    # and not setting grad = 0 in singleStep()
    def setZeroGrads(self):
        # Set the gradients to zero
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
             
    # Call this only after loss.backward() to apply dy/dx to the model parameters
    def singleStep(self):
        count = 0  

        for param in self.model.parameters(): # Loop executed ~37 times
            # these gradients are already computed            
            
            sanitizedGrads = self.getSanitizedGradients(param.grad.data) # Can also use param.grad
            # Update the params, This is what optimizer.step() does
            param.data = param.data - (self.learning_rate * sanitizedGrads)
            
            # Note: In the Medium article, the noise is added after gradient is computed
            # which is at this point. But in Abadi et al. the noise is added before
            # This is equivalent to param.zero_grad() in Optimizer
            #= tensorInit # Reset for next iteration
            count += 1
        
        self.privacyAccountant.computePrivacySpending(self.ε, self.δ)
        #print(count)
    
    def getPrivacySpent(self):
        # Get the privacy spent so far
        eps, delta = self.privacyAccountant.getPrivacySpent()
        return (eps, delta)

    def hasReachedPrivacyLimit(self):
        # Check if the privacy limit has been reached
        eps, delta = self.privacyAccountant.getPrivacySpent()
        if eps > self.max_eps or delta > self.max_delta:
            return True
        return False


        
            
            


    