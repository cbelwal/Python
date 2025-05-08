import torch

# ----------------- Inference -----------------
# These are the main inference functions, as the model object does not have any
#
# N - batch size 
# T - sequence length (number of tokens in a sentence)
# V - vocab size
#
# Refs: 
# https://huggingface.co/blog/how-to-generate
# https://medium.com/@adimodi96/from-logits-to-tokens-9a36feab9cab
# https://stackoverflow.com/questions/78877667/top-p-sampling-not-working-cuda-error-device-side-assert-triggered
#----------------------------------------------
class CCustomInference:
    def __init__(self, model, tokenizer, device, debug=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debug = debug
        

    def log(self, message):
        if self.debug: 
            print(message)
        
    def getGreedySampling(self, relevantOutputs):
        # argmax is greedy search
        # there is no need to take Softmax and then argmax as index will be the same
        predictionId = torch.argmax(relevantOutputs, axis=-1)
        return predictionId
    
    # topP is a nucleus sampling technique, using cumulative probabilities
    # list is first sorted with highest probability first
    # lower topP and lower temperature will result in less randomness    
    def getTopPFilteredIndices(self, logits, topP=1.0):
        sortedLogits, sortedIndices = torch.sort(logits, descending=True)
        cumulativeProbs = torch.cumsum(torch.softmax(sortedLogits, dim=-1), dim=-1)
        sortedIndicesToKeep = sortedIndices[cumulativeProbs <= topP]
        return sortedIndicesToKeep

    # Use topP to filter the logits and then sample from the filtered logits
    # Use softmax to get the probabilities with temperature
    # Use multinomial to get the index of the random sampled token
    # logits shapte is: N x T x V
    def getRandomSampling(self, logits, temperature=1.0, topP=1.0):
        # topK should be less than maxLen
        tensorTemperature = torch.tensor([temperature]).to(self.device)
        tempModifiedLogits = (logits/tensorTemperature).squeeze(0)
        topPFilteredLogitsIndices = self.getTopPFilteredIndices(tempModifiedLogits, topP)
        topPFilteredLogitsValues = tempModifiedLogits[topPFilteredLogitsIndices]
        
        # Map indices in topPFilteredLogitsValues to topPFilteredLogitsIndices
        # create tensor map
        mapIndices = {}
        for i in range(len(topPFilteredLogitsIndices)):
            mapIndices[i] = topPFilteredLogitsIndices[i]     

        # Get the probabilities for topPFilteredLogits
        topPProbabilities = torch.softmax(topPFilteredLogitsValues, dim=-1)
        # multinomial: Returns a tensor where each row contains num_samples indices 
        # sampled from the multinomial 
        # example: topPProbabilities = [0.3, 0.25, 0.15, 0.1], indexed by categories 0,1,2,3
        # num_samples = 5, and value returned is: tensor([3, 2, 3, 1, 0])
        # it means category 3 was sampled 2 times, category 2,1, and 0 were sampled 1 time    
        # if replacement is True any number of samples can be taken
        # A higher temperature will bring probabilities closer to each other, 
        # increasing chances for multinomial to pick from any of those values
        # A lower temperature will bring probabilities further apart,
        # increasing chances for multinomial to pick the highest probability 
        # Note: The further apart the probabilities are, the higher the temperature value should be
        # to get a more uniform distribution of probabilities 
        self.log(f"TopPProbabilities: {topPProbabilities}")
        predictionId = torch.multinomial(topPProbabilities, num_samples=1, replacement = False)
        #TODO: verify tensor is working
        return mapIndices[predictionId.item()].clone().detach()

    def fromBeamSearch(self, relevantOutputs, beamSize=3):
        pass

    def getInferedTokenIds(self, input, temperature=1.0, topP=1.0):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            logits = self.model(input)
        # logits will contain probabilities for each token    
        # output contains the logits
        # get the index for the highest logits for each token
        # input Shape: N x T x V
        # output Shape: N x T x V
        # relevantLogits Shape: N x V
        relevantLogits = logits[:, -1, :] # Get only the last token output
        # relevantLogits has the logits for the last token
        self.log(f"Relevant outputs shape: {relevantLogits.shape}")
        
        if(temperature == 0.0):
            predictionId = self.getGreedySampling(relevantLogits)  
        else:
            predictionId = self.getRandomSampling(relevantLogits, temperature=temperature, topP=topP)
        self.log(f"Inputs Decoded: {self.getDecodedSentence(input)}")
        return predictionId # return as a tensor
        
    def getDecodedSentence(self,tensorInputTokens):
        # Convert to list as self.tokenizer does not use tensors
        inputTokenIds = tensorInputTokens.squeeze(0).tolist()
        return self.tokenizer.decode(inputTokenIds)

    def getInferenceOutput(self,prompt, temperature=0.0, topP=1.0):
        tokenizedPrompt = self.tokenizer.encode(prompt) # will add start and end tokens
        # Remove the SEP Token at the end
        inputTokenIds = tokenizedPrompt[:-1] # Mask is not being considered at this time
        tensorInputTokenIds = torch.tensor(inputTokenIds).unsqueeze(0).to(self.device)
        currentLen = len(inputTokenIds)
        while(currentLen < self.tokenizer.getMaxLen()):
            currentLen += 1
            predTokenId = self.getInferedTokenIds(tensorInputTokenIds, temperature=temperature, topP=topP)
            tensorInputTokenIds = torch.hstack((tensorInputTokenIds, predTokenId.view(1, 1)))
            if predTokenId == self.tokenizer.sepTokenId:
                break
        return self.getDecodedSentence(tensorInputTokenIds)    
    
if __name__ == "__main__":
    sampleLogits = torch.tensor([[0.15, 0.1, 0.3, 0.25]]) # idx 2 (value .3) is the highest 
    #sampleLogits = torch.tensor([[0.15, 0.1, 0.0000004, 0.000003]])
    #sampleLogits = torch.tensor([[0.0659, 0.0625, 0.0544, 0.0479, 0.0463, 0.0463, 0.0461, 0.0450, 0.0408,
    #    0.0365, 0.0352, 0.0323, 0.0257, 0.0234, 0.0208, 0.0200, 0.0199, 0.0198,
    #    0.0183, 0.0176, 0.0165, 0.0162, 0.0156, 0.0153, 0.0148, 0.0142, 0.0141,
    #    0.0137, 0.0132, 0.0129, 0.0128, 0.0124, 0.0123, 0.0116, 0.0113, 0.0103,
    #    0.0090, 0.0089, 0.0088, 0.0084, 0.0083, 0.0075, 0.0073]])
    inferenceObj = CCustomInference(None, None,None, debug=True)
    print(f"Greedy Sampling: {inferenceObj.getGreedySampling(sampleLogits)}") # output 2
    # Will return sorted list of index till cumpub is reached
    # output 2,3,0 for values 0.3, 0.25, 0.15
    print(f"TopP Filtered Indices: {inferenceObj.getTopPFilteredIndices(sampleLogits, topP=.8)}") 
    print(f"Random Sampling with (Temp = 1.0, topP = 1.0): {inferenceObj.getRandomSampling(sampleLogits, temperature=1.0, topP=1.0)}")
    print(f"Random Sampling with (Temp = 1.0, topP = 0.6): {inferenceObj.getRandomSampling(sampleLogits, temperature=1.0, topP=1.0)}")
    print(f"Random Sampling with (Temp = .5, topP = 1.0): {inferenceObj.getRandomSampling(sampleLogits, temperature=0.5, topP=1.0)}")
    print(f"Random Sampling with (Temp = .5, topP = 0.6): {inferenceObj.getRandomSampling(sampleLogits, temperature=0.5, topP=0.6)}")