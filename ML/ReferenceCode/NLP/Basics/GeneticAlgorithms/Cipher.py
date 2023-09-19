import string
import random
import numpy as np
import re

#--------- Build Mapping
letters1 = list(string.ascii_lowercase) #return a-z
letters2 = list(string.ascii_lowercase) #return a-z

#Do a random shuffle of letters2
random.shuffle(letters2)
mapping = {}
reverseMapping = {}
for i in range(len(letters1)):
    mapping[letters1[i]] = letters2[i]
    reverseMapping[letters2[i]] = letters1[i]

#print(mapping)
#Only sender/receiver know true mapping
#------------ Mapping done
#---- Build Markov Model Matrix ------------------------------
V = 26
A = np.ones((V, V)) #26x26 matrixof char prob
pi = np.zeros(V) #np.zeros(V)

input_file= 'c:/users/chaitanya belwal/.datasets/nlp/moby_dick.txt'
#input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test.txt'
#input_file= 'c:/users/chaitanya belwal/.datasets/nlp/test1.txt'
#textToEncrypt = "life"
textToEncrypt = "This is a sample text that will be encoded using our default mapping"
#textToEncrypt = "I then lounged down the street and found"

regex = re.compile('[^a-zA-Z]') #Define allowed regex
for line in open(input_file,encoding="utf8"):
    line = line.rstrip().lower()
    if line:
        line = regex.sub(' ',line) #Replace words matching regex with ' '

        tokens = line.split(' ') #split into words
        for token in tokens:
            token = token.lower()
            lastC = -1
            for c in token:
                ci = ord(c) - 97
                if(lastC < 0):
                    pi[ci] += 1
                else:
                    A[lastC,ci] += 1
                lastC = ci    

#Convert to prob. based on sum. Use numpy to compute sum
pi = pi/pi.sum()
A = A/A.sum(axis=1, keepdims=True)

logpi = np.log(pi)
logA = np.log(A)

#for i in range(26):
#    A[i] = A[i]/A[i].sum(axis=1, keepdims=True)


#Now we have the Markov Matrix -----------------------
#Build encoded text for each sentence
#Use mapping
def getSentenceOnMapping(stringToEncode,mapping):
    stringToEncode = stringToEncode.lower()
    stringToEncode = regex.sub(' ', stringToEncode)
    encrypted = []
    for c in stringToEncode:
        if c != ' ':
            encrypted.append(mapping[c])
        else:
            encrypted.append(' ')

    encStr = ""
    for c in encrypted:
        encStr += c
    
    return encStr



encStr = getSentenceOnMapping(textToEncrypt,mapping)
print("Encrypted String:",encStr)
print("Decrypted String:",getSentenceOnMapping(encStr,reverseMapping))

#---- Start to find mapping that will decrypt it
def getMLEWord(word):
    lastC = -1
    mle = 0
    for c in word:
        ci = ord(c) - 97 #decrypted char
        if lastC==-1:
            mle =  logpi[ci] #logPi[ci]
        else:
            mle += logA[lastC,ci] #logA[lastC,ci]
        lastC = ci
    return mle

def getMLESentence(sentence): #Should be decrypted sentence
    words = sentence.split(' ')
    mle = 0
    for word in words:
        mle += getMLEWord(word)
    return mle

def getMappingsBySingleSwap(mapping,n):
    mappings = [] #List of dictionaries
    for i in range(n):
        newMap = mapping.copy()
        num1 = np.random.randint(26)
        num2 = np.random.randint(26)
        #Swap nums[0] and nums[1]
        chars = [0] * 2
        chars[0] = chr(97 + int(num1))
        chars[1] = chr(97 + int(num2))
        temp = newMap[chars[0]]
        newMap[chars[0]] = newMap[chars[1]]
        newMap[chars[1]] = temp
        mappings.append(newMap)
    #print(mappings)
    return mappings


def geneticLoop(parentMaps,sentence,generations):
    #Each parent has 3 child, one of itself and two modified
    cParentMaps = parentMaps.copy()
    bestScore = (float('-inf'),{})
    for i in range(generations):
        for parentMap in parentMaps:
            tMaps = getMappingsBySingleSwap(parentMap, 5) #Get 3 children
            cParentMaps += tMaps

        #Now compute MLEs for all children
        allScores = []
        for parentMap in cParentMaps:
            decStr = getSentenceOnMapping(sentence,parentMap)
            mle = getMLESentence(decStr)
            #if(mle > 1.14): For debugging
            #    print("yo")
            #    mle = getMLESentence(decStr)
            allScores.append ((mle,parentMap))
            
        
        #Clean next generation
        cParentMaps = []
        #determine top 5 mles
        allScores.sort(key=lambda x:x[0],reverse=True)
        
        itr = 0
        for tuple in allScores:
            if tuple[0] > bestScore[0]:
                bestScore = tuple
            itr += 1
            cParentMaps.append(tuple[1])
            if(itr == 5):
                break
       
        parentMaps = cParentMaps.copy() #Required in the main loop to iterate
        if(i % 50 == 0): 
            print("Iter:",i," best mle:",bestScore[0])

    
    print("*** Final Iter:",i," best mle:",bestScore[0])
    return bestScore[1]

print("*** MLE for original sentence",getMLESentence(textToEncrypt))
#exit()
#Build initial parents will full randomness
startParentMaps = []
for _ in range(40):
    tempMapping = {}
    random.shuffle(letters2)
    for i in range(len(letters1)):
        tempMapping[letters2[i]] = letters1[i] #axmmy -> hello
    startParentMaps.append(tempMapping)

print("Starting genetic loop ...")
bestMap = geneticLoop(startParentMaps,encStr,1000) #1000 generations
 

#Final string with best mapping
print("Reverse Mapping",bestMap)
print("Decrypted String:",getSentenceOnMapping(encStr,bestMap))
#print("Decrypted String:",getSentenceOnMapping(encStr,reverseMapping))


