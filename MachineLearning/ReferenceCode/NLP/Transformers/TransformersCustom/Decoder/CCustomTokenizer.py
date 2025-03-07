# Simple work cases tokenider
class CCustomTokenizer:
    def __init__(self, fileName):
        self.fileName = fileName
        self.wordToTokenId = {}
        self.tokenIdToWord = {}
        self.tokenIdxToTokenId = {}
        # Do not set this to higher values as the vocab size in nn.embeddings layer
        # is based on the index
        self.regularTokenStartIdx = 5 # Anything below this is reserved for special tokens
        self.maxLen = 0
        self.maxTokenId = 0
        self.allLines = []
        self.assignSpecialTokens()
        self.generateTokens()
        
    def read_file(self):
        with open(self.fileName, 'r') as file:
            return file.readlines()
    
    def assignSpecialTokens(self):
        self.startToken = 1
        self.sepToken = 2
        self.padToken = 0
        self.unkToken = 3
        self.wordToTokenId["CLS"] = self.startToken
        self.tokenIdToWord[self.startToken] = "CLS"
        self.wordToTokenId["SEP"] = self.sepToken
        self.tokenIdToWord[self.sepToken] = "SEP"
        self.wordToTokenId["PAD"] = self.padToken
        self.tokenIdToWord[self.padToken] = "PAD"
        self.wordToTokenId["UNK"] = self.unkToken
        self.tokenIdToWord[self.unkToken] = "UNK"

    def getPadTokenId(self):
        return self.padToken

    def getSepTokenId(self):
        return self.sepToken

    def generateTokens(self):
        allRawLines = self.read_file()
        tokenId = self.regularTokenStartIdx + 1
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
                    self.tokenIdxToTokenId[idx] = tokenId
                    idx += 1
                    tokenId += 1
        self.maxTokenId = tokenId

    def getMaxLen(self):
        return self.maxLen + 2 # For CLS and SEP tokens

    def getMaxTokenId(self):
        return self.maxTokenId

    def getVocabSize(self):
        return len(self.wordToTokenId)

    def getTokenIdforIdx(self, idx):
        return self.tokenIdxToTokenId[idx]

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
    
    def encodeTokenizedSentence(self, sentence,maxLen=None):
        if maxLen is None:
            maxLen = self.getMaxLen()
        words = sentence.split(' ')
        tokens = []
        tokens.append(self.getTokenIdForWord("CLS"))
        for word in words:
            tokens.append(self.getTokenIdForWord(word))
        tokens.append(self.getTokenIdForWord("SEP"))
        
        # Pad the sentence
        if len(tokens) < maxLen:
            for i in range(maxLen - len(tokens)):
                tokens.append(self.getTokenIdForWord("PAD"))

        return tokens
    
    def decodeTokenizedSentence(self, tokens):
        words = []
        for token in tokens:
            words.append(self.getWordForTokenId(token))
        return ' '.join(words)

    def getAllTrainingRows(self,maxLen=None):
        if maxLen is None:
            maxLen = self.getMaxLen()
        allRows = []
        for line in self.allLines:
            tokens = self.encodeTokenizedSentence(line,maxLen)
            allRows.append(tokens)
        return allRows


if __name__=="__main__":
    tokenizer = CCustomTokenizer("./data/SampleSentencesCorrected.txt")
    print("Total tokens: ", tokenizer.getVocabSize())
    print("MaxLen: ", tokenizer.getMaxLen())
    sentence = "the cat is good"
    tokens = tokenizer.encodeTokenizedSentence(sentence)
    print("Tokens: ", tokens)
    decodedSentence = tokenizer.decodeTokenizedSentence(tokens)
    print("Decoded sentence: ", decodedSentence)
    print("All training data:\n ", tokenizer.getAllTrainingRows())