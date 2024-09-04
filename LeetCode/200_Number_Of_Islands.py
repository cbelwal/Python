from typing import Dict, List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        def rec(r,c):
            if(r== -1 or r== m or c==-1 or c==n):
                return True
            if(grid[r][c]=="0"): #Reached water
                return True
            if(visited[r][c] == "X"):
                return True
           
            visited[r][c] = "X" #Dont mark "0"
            
            flag = True
            for dir in dirs:
                val = rec(r+dir[0],c+dir[1])
                flag = flag or val
            return flag

        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        
        m = len(grid)
        n = len(grid[0])
        #arr = [["" for j in range(n)] for i in range(m)]
        visited = [[""]*n for i in range(m)]
        count = 0

        for i in range(m):
            for j in range(n):
                if(grid[i][j] == "1" and visited[i][j] != "X"):
                    flag = rec(i,j)
                    if(flag):
                        count += 1
        return count
    
solution = Solution()
array = [["1","1","1","1","0"],["1","1","0","1","0"],
["1","1","0","0","0"],["0","0","0","0","0"]]
val = solution.numIslands(array)
print(val)