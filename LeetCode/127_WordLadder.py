from typing import Deque, List
from collections import deque

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not endWord in wordList:
            return 0

        visited = {}


        #Build Adjacency List
        L = len(wordList[0])
        wordList.append(beginWord)
        dictAll = {}
        for w in wordList:
            for i in range(0,L):
                tmpW = w[:i] + '*' + w[i+1:]
                if(tmpW in dictAll):
                    dictAll[tmpW].append(w)
                else:
                    tmpL = [w]
                    dictAll[tmpW] = tmpL

        queue = Deque()
        queue.append((beginWord,1))
        visited[beginWord] = 1
        while(len(queue) > 0):
            tuple = queue.popleft()

            for i in range(0,L):
                tmpW = tuple[0][:i] + '*' + tuple[0][i+1:]
                for w in dictAll[tmpW]:
                    if(w == endWord):
                        return tuple[1] + 1
                    if(not w in visited):
                        queue.append((w,tuple[1]+1))
                        visited[w] = 1
        return 0


solution = Solution()
#wordList = ["hot","dot","dog","lot","log","cog"]
#val = solution.ladderLength("hit","cog",wordList)
wordList = ["ted","tex","red","tax","tad","den","rex","pee"]
val = solution.ladderLength("red","tax",wordList)
print(val)