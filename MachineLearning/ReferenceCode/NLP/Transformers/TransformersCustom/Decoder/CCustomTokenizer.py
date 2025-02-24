# Simple work cases tokenider
class CCustomTokenizer:
    def __init__(self, fileName):
        self.fileName = fileName
        self.allLines = self.read_file()
        self.wordToTokenId = {}
        self.tokenIdToWord = {}
        self.specialTokenEndIdx = 4
        self.assignSpecialTokens()
        self.generateTokens()

    def read_file(self):
        with open(self.fileName, 'r') as file:
            return file.readlines()
    
    def assignSpecialTokens(self):
        self.startToken = 0
        self.endToken = 1
        self.wordToTokenId["CLS"] = self.startToken
        self.tokenIdToWord[self.startToken] = "CLS"
        self.wordToTokenId["SEP"] = self.endToken
        self.tokenIdToWord[self.endToken] = "SEP"

    def generateTokens(self):
        idx = self.specialTokenEndIdx + 1
        for line in self.allLines:
            if line.strip() == "":
                continue
            
            # Remove special characters
            line = line.replace("\n", "")
            line = line.replace("\r","")
            line = line.replace(",","")
            line = line.replace(".","") # Encoder for single word sentences
            line = line.lower()

            words = line.split(' ')
            for word in words:
                if word not in self.wordToTokenId:
                    self.wordToTokenId[word] = idx
                    self.tokenIdToWord[idx] = word
                    idx += 1

    def getCountOfTokens(self):
        return len(self.wordToTokenId)

    def getWordForTokenId(self, tokenId):
        return self.tokenIdToWord[tokenId]
    
    def getTokenIdForWord(self, word):
        return self.wordToTokenId[word]

    def getTokenizedSentence(self, sentence):
        words = sentence.split(' ')
        tokens = []
        tokens.append(self.getTokenIdForWord("CLS"))
        for word in words:
            tokens.append(self.getTokenIdForWord(word))
        tokens.append(self.getTokenIdForWord("SEP"))
        return tokens
    
    def decodeTokenizedSentence(self, tokens):
        words = []
        for token in tokens:
            words.append(self.getWordForTokenId(token))
        return ' '.join(words)


if __name__=="__main__":
    tokenizer = CCustomTokenizer("./data/SampleSentencesCorrected.txt")
    print("Total tokens: ", tokenizer.getCountOfTokens())
    sentence = "the cat is good"
    tokens = tokenizer.getTokenizedSentence(sentence)
    print("Tokens: ", tokens)
    decodedSentence = tokenizer.decodeTokenizedSentence(tokens)
    print("Decoded sentence: ", decodedSentence)