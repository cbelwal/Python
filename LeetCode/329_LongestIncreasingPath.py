from typing import Dict, List

class Solution:
    
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        #Unique value = 0,1,2: colcount * row + col
        self.memo = {}
        self.dirs = [[0,1],[0,-1],[1,0],[-1,0]]
        self.m = len(matrix) #m = rows
        self.n = len(matrix[0]) #n = cols
        maxPath = 0
        for r in range(0,self.m):
            for c in range (0,self.n):
                maxPath = max(maxPath,self.getPathLength(matrix,r,c))
        return maxPath

    def getPathLength(self, matrix: List[List[int]], row:int,col:int)->int:
            idx = self.n * row + col
            
            if(idx in self.memo):
                return self.memo[idx]

            maxL = 1
            for dir in self.dirs:
                if (row+dir[0] < 0 or row+dir[0] >= self.m): continue
                if (col+dir[1] < 0 or col+dir[1] >= self.n): continue

                if(matrix[row + dir[0]][col+dir[1]] > matrix[row][col]):
                    idx = self.n * (row + dir[0]) + (col + dir[1])
                    if(idx in self.memo):
                        length = self.memo[idx]
                    else:
                        length = self.getPathLength(matrix,row +dir[0],col+dir[1])+1
                    self.memo[idx] = length
                    maxL = max(maxL, length)

            return maxL

solution = Solution()
#matrix = [[9,9,4],[6,6,8],[2,1,1]]
#matrix = [[3,4,5],[3,2,6],[2,2,1]]
#matrix = [[1]]
matrix = [[0,1,2,3,4,5,6,7,8,9],[19,18,17,16,15,14,13,12,11,10],[20,21,22,23,24,25,26,27,28,29],[39,38,37,36,35,34,33,32,31,30],[40,41,42,43,44,45,46,47,48,49],[59,58,57,56,55,54,53,52,51,50],[60,61,62,63,64,65,66,67,68,69],[79,78,77,76,75,74,73,72,71,70],[80,81,82,83,84,85,86,87,88,89],[99,98,97,96,95,94,93,92,91,90],[100,101,102,103,104,105,106,107,108,109],[119,118,117,116,115,114,113,112,111,110],[120,121,122,123,124,125,126,127,128,129],[139,138,137,136,135,134,133,132,131,130],[0,0,0,0,0,0,0,0,0,0]]
val = solution.longestIncreasingPath(matrix)
print(val)
