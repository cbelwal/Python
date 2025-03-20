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

    def log(self, message):
        if self.debug: 
            print(message)
        
    def getGreedySampling(self, relevantOutputs):
        # argmax is greedy search
        # there is no need to take Softmax and then argmax as index will be the same
        predictionId = torch.argmax(relevantOutputs, axis=-1)
        return predictionId
    
    # topP is a nucleus sampling technique, using cumulative probabilities
    def getTopPFilteredIndices(self, logits, topP=1.0):
        sortedLogits, sortedIndices = torch.sort(logits, descending=True)
        cumulativeProbs = torch.cumsum(torch.softmax(sortedLogits, dim=-1), dim=-1)
        sortedIndicesToKeep = sortedIndices[cumulativeProbs <= topP]
        return sortedIndicesToKeep

    def getRandomSampling(self, logits, temperature=1.0, topP=1.0):
        # topK should be less than maxLen
        tensorTemperature = torch.tensor([temperature]).to(self.device)
        tempModifiedLogits = logits/tensorTemperature
        topPFilteredLogitsIndices = self.getTopPFilteredIndices(tempModifiedLogits, topP)
        topPFilteredLogitsValues = logits[topPFilteredLogitsIndices]
        # Get the probabilities for topPFilteredLogits
        topPProbabilities = torch.softmax(topPFilteredLogitsValues, dim=-1)
        # multinomial: Returns a tensor where each row contains num_samples indices 
        # sampled from the multinomial 
        predictionId = torch.multinomial(topPProbabilities, num_samples=1)
        return predictionId
    
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
        # relevantOutputs Shape: N x V
        relevantLogits = logits[:, -1, :] # Get only the last token output
        # relevantLogits has the logits for the last token
        self.log("Relevant outputs shape",relevantLogits.shape)
        
        if(temperature == 0.0):
            predictionId = self.fromGreedySampling(relevantLogits)  
        else:
            predictionId = self.fromRandomSampling(relevantLogits, temperature=temperature, topP=topP)
        self.log("Inputs Decoded",self.getDecodedSentence(input))
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
        len = 2
        while(len < self.tokenizer.getMaxLen()):
            len += 1
            predTokenId = self.getInferedTokenIds(self.model, tensorInputTokenIds, temperature=temperature)
            tensorInputTokenIds = torch.hstack((tensorInputTokenIds, predTokenId.view(1, 1)))
            if predTokenId == self.tokenizer.sepTokenId:
                break
        return self.getDecodedSentence(tensorInputTokenIds)    
    
if __name__ == "__main__":
    sampleLogits = torch.tensor([0.15, 0.1, 0.3, 0.25]) # idx 2 (value .3) is the highest 
    inferenceObj = CCustomInference(None, None,None, debug=True)
    print(f"Greedy Sampling: {inferenceObj.getGreedySampling(sampleLogits)}") # output 2
    # Will return sorted list of index till cumpub is reached
    # output 2,3,0 for values 0.3, 0.25, 0.15
    #print(f"TopP Filtered Indices: {inferenceObj.getTopPFilteredIndices(sampleLogits, topP=.8)}") 
    print(f"Random Sampling with Temp = 1.0, topP = 1.0: {inferenceObj.getRandomSampling(sampleLogits, temperature=1.0, topP=1.0)}")
    pass