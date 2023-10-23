from typing import List

class Solution:
    def getNewFormat(self,oldValue,newValue):
        if(oldValue == newValue):
            return oldValue
        if(oldValue == 0):
            return -1 #newValue == 1
        else: #oldValue = 1
            return -2 #newValue = 0

    #0->1, -1
    #0->0, 0
    #1->0, -2
    #1->1, 1    
    def getOldValue(self,actualNew):
        if(actualNew >= 0):#Not changed
            return actualNew
        if(actualNew == -2): 
            return 1 #Mod value =-2, actual value 0, oldValue = 1
        else:
            return 0 #Mod value =-1, actual value 1, oldValue = 0
        
    def getTransformValue(self,actualNew):
        if(actualNew >= 0):#Not changed
            return actualNew
        if(actualNew == -2): 
            return 0 #Mod value =-2, actual value 0, oldValue = 1
        else:
            return 1 #Mod value =-1, actual value 1, oldValue = 0

    def getNewState(self,row,col): 
        if(row == 0 and col == 2):
            pass
        dirs = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(-1,1),(1,-1)]
        count = 0 #Count of 1 neighbors
        for dir in dirs:
            if(row+dir[0] >= 0 and row + dir[0] < self.rows):
                if (col + dir[1] >= 0 and col+dir[1] < self.cols):
                    if(self.getOldValue(self.board[row+dir[0]][col+dir[1]]) == 1):
                        count +=1
                    
        if count < 2:
            return self.getNewFormat(self.board[row][col],0)
        if count == 3:
            return self.getNewFormat(self.board[row][col],1)
        if count > 3:
            return self.getNewFormat(self.board[row][col],0)
        else: #Make no change
            return self.board[row][col] 

    # T: O(m.n)
    # S: O(1)
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        #0->1, -1
        #1->0, -2
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

        # O(m.n)
        for r in range(0,self.rows):
            for c in range(0,self.cols):
                self.board[r][c] = self.getNewState(r,c)

        print(board)
        #Now revert to existing value
        for r in range(0,self.rows):
            for c in range(0,self.cols):
                self.board[r][c] = self.getTransformValue(self.board[r][c])

solution = Solution()
board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
solution.gameOfLife(board)
print(board)