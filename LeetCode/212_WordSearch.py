#Used special queue
from typing import Dict, List

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        self.board = board
        self.m = len(board)
        self.n = len(board[0])
        
        self.result = {}
        trie = self.buildTrie(words)

        tmpBoard = [[False for x in range(self.n)] for y in range(self.m)]
        for row in range(0,self.m):
            for col in range (0,self.n):
                self.doDFS(row,col,trie,tmpBoard)
            
        ret = []
        for k in self.result.keys(): 
            ret.append(k)
        return ret
        
    
    def doDFS(self,row, col, trieNode:Dict, tmpBoard):
        dirs = [[0,1],[1,0],[-1,0],[0,-1]]
        letter  = self.board[row][col]
        
        if letter not in trieNode:
            return
        currentNode = trieNode[letter]

        if('$' in currentNode):
            self.addWordToResult(currentNode['$'])
            if(len(currentNode) == 1): #leaf node
                trieNode.pop(letter)

        tmpBoard[row][col] = True
        for d in dirs:
            if(row + d[0] >= 0 and row + d[0] < self.m 
            and col + d[1] >= 0 and col + d[1]< self.n and 
            tmpBoard[row+d[0]][col+d[1]]==False):
                self.doDFS(row+d[0],col+d[1],currentNode,tmpBoard)
        tmpBoard[row][col] = False


    def addWordToResult(self,mainS:str):
        if(mainS not in self.result):
            self.result[mainS] = 1
        
    def buildTrie(self,words: List[str])->Dict:
        rootTrie = {} 
        node = rootTrie
        for word in words:
            node = rootTrie
            for c in word:
                node = node.setdefault(c,{}) 
            node['$'] = word
        return rootTrie

solution = Solution()
board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
wordList = ["oath","pea","eat","rain"]
#board = [["a","b","c","e"],["x","x","c","d"],["x","x","b","a"]]
#wordList = ["abc","abcd"]
#board = [["a","a"]]
#wordList = ["aaa"]
val = solution.findWords(board,wordList)
print(val)