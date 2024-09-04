import math

class Solution:
    def __init__(self):
        self.squares = []
        
    def numSquares(self, n: int) -> int:
        self.genSquares(n)
        dp = [float('inf')] * (n+1)
        dp[0] = 0
        
        for i in range(1,n+1):
            for s in range(1,len(self.squares)):
                if i < s:
                    break
                dp[i] = min(dp[i],dp[i-self.squares[s]]+1)
        
        return dp[n]

    # Returns list of squares
    def genSquares(self,n):
        self.squares = []
        self.squares.append(0)
        for i in range(1,int(math.sqrt(n))+1):
            self.squares.append(i**2)

solution = Solution()
val = solution.numSquares(12)
#val = solution.numSquares(22)
#val = solution.numSquares(13)
#val = solution.numSquares(2)
#val = solution.numSquares(19)
#val = solution.numSquares(1)

print(val)
        

        