'''
Write atleast 200 possible sentences that contain only the following words

the
cat
is
jumps
house
over
good
black
it
a

Each sentence is at least has 3 words and can contain each of the words multiple times as long as the sentence is valid.
Do not make a sentence that has a work that does not belong to the list above. 

'''

def read_file(fileName):
    with open(fileName, 'r') as file:
        return file.readlines()

# Write all the dictionary lines to a file
def write_file(fileName, dictLines):
    with open(fileName, 'w') as file:
        for line in dictLines:
            file.write(line)


if __name__ == "__main__":
    fileName = "./data/SampleSentencesRaw.txt"
    outFileName = "./data/SampleSentencesCorrected.txt"

    allLines = read_file(fileName)
    dictLines = {}
    for line in allLines:
        if line.strip() == "":
            continue
        if line not in dictLines:
            dictLines[line] = 1
    
    print("Unique lines: ", len(dictLines)) 
    write_file(outFileName, dictLines)
    print("Done writing to file: ", outFileName)
    