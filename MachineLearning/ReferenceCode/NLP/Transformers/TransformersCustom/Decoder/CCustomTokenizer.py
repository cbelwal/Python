# Simple work cases tokenider
class CCustomTokenizer:
    def __init__(self, fileName):
        self.fileName = fileName
        self.wordToTokenId = {}
        self.tokenIdToWord = {}
        self.tokenIdxToTokenId = {}
        # Do not set this to higher values as the vocab size in nn.embeddings layer
        # is based on the index
        self.maxLen = 0
        self.maxTokenId = 0
        self.allLines = []
        self.generateTokens()
        
    def read_file(self):
        with open(self.fileName, 'r') as file:
            return file.readlines()
    
    def assignSpecialTokens(self):
        self.padTokenId = 0;self.padToken = "<PAD>"
        self.startTokenId = 1; self.startToken= "<CLS>"
        self.sepTokenId = 2; self.sepToken = "<SEP>"    
        self.unkTokenId = 3; self.unkToken = "<UNK>"
        self.tokenIdToWord[self.startTokenId] = self.startToken
        self.wordToTokenId[self.tokenIdToWord[self.startTokenId]] = self.startTokenId
        self.tokenIdToWord[self.sepTokenId] = self.sepToken
        self.wordToTokenId[self.tokenIdToWord[self.sepTokenId]] = self.sepTokenId
        self.tokenIdToWord[self.padTokenId] = self.padToken
        self.wordToTokenId[self.tokenIdToWord[self.padTokenId]] = self.padTokenId
        self.tokenIdToWord[self.unkTokenId] = self.unkToken
        self.wordToTokenId[self.tokenIdToWord[self.unkTokenId]] = self.unkTokenId
        

    def getPadTokenId(self):
        return self.padTokenId

    def getSepTokenId(self):
        return self.sepTokenId

    def generateTokens(self):
        self.assignSpecialTokens()
        tokenId = len(self.tokenIdToWord) # start right after special tokens
        allRawLines = self.read_file()
         
        idx = 0 # Mainly for use to get the logits index
        for line in allRawLines:
            if line.strip() == "":
                continue
            # Remove special characters
            line = line.replace("\n", "")
            line = line.replace("\r","")
            line = line.replace(",","")
            line = line.replace(".","") # Encoder for single word sentences
            line = line.lower()

            self.allLines.append(line)
            words = line.split(' ')
            if(len(words) > self.maxLen):
                self.maxLen = len(words)

            for word in words:
                if word not in self.wordToTokenId:
                    self.wordToTokenId[word] = tokenId
                    self.tokenIdToWord[tokenId] = word
                    idx += 1
                    tokenId += 1
            self.maxTokenId = tokenId     
        
        # assign the tokenIdx to tokenId for all tokens
        # its important to do it at this stage after both special and normal
        # tokens are assigned
        idx = 0
        for tokenId in self.tokenIdToWord:
            self.tokenIdxToTokenId[idx] = tokenId
            idx += 1
        

    def getMaxLen(self):
        return self.maxLen + 2 # For CLS and SEP tokens

    def getMaxTokenId(self):
        return self.maxTokenId

    def getVocabSize(self):
        return len(self.wordToTokenId)

    #def getTokenIdforIdx(self, idx):
    #    return self.tokenIdxToTokenId[idx]

    def getWordForTokenId(self, tokenId):
        return self.tokenIdToWord[tokenId]
    
    def getTokenIdForWord(self, word):
        return self.wordToTokenId[word]

    '''
    Takes in a list of idxs and returns list of token ids
    
    The idxs comes from the logits of the model
    '''
    def getTokenIdsForIdxs(self, idxs):
        tokenIds = []
        for idx in idxs:
            tokenIds.append(self.getTokenIdforIdx(idx))
        return tokenIds
    
    # Return tokenid encoded sentences with special tokens
    def encode(self, sentence,addPadding=False,maxLen=None):
        
        words = sentence.split(' ')
        tokens = []
        tokens.append(self.getTokenIdForWord(self.startToken))
        for word in words:
            tokens.append(self.getTokenIdForWord(word.lower()))
        tokens.append(self.getTokenIdForWord(self.sepToken))
        
        # Pad the sentence
        if(addPadding):
            if maxLen is None:
                maxLen = self.getMaxLen()
            if len(tokens) < maxLen:
                for i in range(maxLen - len(tokens)):
                    tokens.append(self.getTokenIdForWord(self.padToken))

        return tokens
    
    def decode(self, tokens):
        words = []
        for token in tokens:
            words.append(self.getWordForTokenId(token))
        return ' '.join(words)

    def getAllTrainingRows(self,maxLen=None):
        if maxLen is None:
            maxLen = self.getMaxLen()
        allRows = []
        for line in self.allLines:
            tokens = self.encode(line,maxLen)
            allRows.append(tokens)
        return allRows


if __name__=="__main__":
    tokenizer = CCustomTokenizer("./data/SampleSentencesCorrected.txt")
    print("Total tokens: ", tokenizer.getVocabSize())
    print("MaxLen: ", tokenizer.getMaxLen())
    sentence = "the cat is good"
    tokens = tokenizer.encode(sentence)
    print("Tokens: ", tokens)
    decodedSentence = tokenizer.decode(tokens)
    print("Decoded sentence: ", decodedSentence)
    print("All training data:\n ", tokenizer.getAllTrainingRows())