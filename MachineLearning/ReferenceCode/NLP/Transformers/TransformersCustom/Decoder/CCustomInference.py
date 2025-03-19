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
    #----------------------------------------------

class CCustomInference:
    def __init__(self, model, tokenizer, device, debug=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def log(self, message):
        if self.debug: 
            print(message)
        
    def fromGreedySampling(self, relevantOutputs):
        # argmax is greedy search
        # there is no need to take Softmax and then argmax as index will be the same
        predictionId = torch.argmax(relevantOutputs, axis=-1)
        return predictionId
    
    def fromRandomSampling(self, relevantOutputs, temperature=1.0, topK=0):
        # topK should be less than maxLen
        tensorTemperature = torch.tensor([temperature]).to(self.device)
        tempModifiedRelevantOutputs = torch.round(torch.exp(relevantOutputs/(tensorTemperature)) / 
                                        torch.sum(torch.exp(relevantOutputs/(tensorTemperature))), decimals=4)
        predictionId = torch.multinomial(relevantOutputs, num_samples=1)
        return predictionId
    
    def fromBeamSearch(self, relevantOutputs, beamSize=3):
        pass


    def getInferedTokenIds(self, input, temperature=0.0, debug=False):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            outputs = self.model(input)
        # logits will contain probabilities for each token    
        # output contains the logits
        # get the index for the highest logits for each token
        # input Shape: N x T x V
        # output Shape: N x T x V
        # relevantOutputs Shape: N x V
        tensorTemperature = torch.tensor([temperature + 1]).to(self.device) # add 1 to prevent div by zero
        relevantOutputs = outputs[:, -1, :] # Get only the last token output
        # relevantOutpurs has the logits for the last token
        #print("Relevant outputs shape",relevantOutputs.shape)
        # if using temperature we are not using the highest logits
        tempModifiedRelevantOutputs = torch.round(torch.exp(relevantOutputs/(tensorTemperature)) / 
                                        torch.sum(torch.exp(relevantOutputs/(tensorTemperature))), decimals=4)
        #predictionId = torch.multinomial(relevantOutputs, num_samples=1) # 1
        #print("Shape:", tempModifiedRelevantOutputs.shape)
        print(tempModifiedRelevantOutputs)
        # argmax is greedy search
        predictionId = torch.argmax(tempModifiedRelevantOutputs, axis=-1) # 1
        print("Top values:",torch.topk(tempModifiedRelevantOutputs.flatten(), 3))
        #print(f"Max Index:{predictionId}, value:{tempModifiedRelevantOutputs[predictionId]}")
        #predictionIds = torch.argmax(outputs, axis=-1)
        #print("Inputs",input)
        if debug:print("Inputs Decoded",self.getDecodedSentence(input))
        #print("Pred ids",predictionIds)
        #print("Preds ids decoded:", getDecodedSentence(predictionIds))
        #print("Prediction Id shape:", predictionId.shape) # torch.Size([1, 12])
        
        return predictionId # return as a tensor
        
    def getDecodedSentence(self,tensorInputTokens):
        # Convert to list as self.tokenizer does not use tensors
        inputTokenIds = tensorInputTokens.squeeze(0).tolist()
        return self.tokenizer.decode(inputTokenIds)

    def getInferenceOutput(self,prompt, temperature=0.0):
        tokenizedPrompt = self.tokenizer.encode(prompt) # will add start and end tokens
        # Remove the SEP Token at the end
        inputTokenIds = tokenizedPrompt[:-1] # Mask is not being considered at this time
        tensorInputTokenIds = torch.tensor(inputTokenIds).unsqueeze(0).to(device)
        len = 2
        while(len < self.tokenizer.getMaxLen()):
            len += 1
            predTokenId = getInferedTokenIds(model, tensorInputTokenIds, temperature=temperature)
            tensorInputTokenIds = torch.hstack((tensorInputTokenIds, predTokenId.view(1, 1)))
            break
            if predTokenId == self.tokenizer.sepTokenId:
                break
        return getDecodedSentence(tensorInputTokenIds)    