from typing import Dict, List

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        #["wrt","wrf","er","ett","rftt"]
        #Reverse Adj List
        revAdjList = {c: [] for word in words for c in word}

        for word1,word2 in zip(words,words[1:]): #O(n.maxL)
            for a,b in zip(word1,word2):
                if(word2 in word1):
                    return "" #is prefix
                if (a != b):           
                    revAdjList[b].append(a) # Reverse Adjanceny
                    break #Skip in word1, word2 loop                         
                
        #Start DFS
        output = []
        seen = {}
        def DFS(node):
            if(node in seen):
                return seen[node]
            
            seen[node] = False
            for n in revAdjList[node]:
                ret = DFS(n)
                if(not ret):
                    return False

            output.append(node)
            seen[node] = True
            return True

        if not all(DFS(n) for n in revAdjList):
            return ""

        return "".join(output)

solution = Solution()
val = solution.alienOrder(["wrt","wrf","er","ett","rftt"])
print(val)