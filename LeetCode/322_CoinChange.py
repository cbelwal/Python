from typing import List
from functools import lru_cache

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        self.coins = coins
        result = self.rec(amount)
        return result-1 if result >= 0 else -1 
        
    # Build a Tree with parent as the remainder
    # lru_cache will act as memo table storing the remainders 
    @lru_cache(None)
    def rec(self, rem):
        if(rem < 0):
            return -1 #No coin will fit
        if(rem == 0):
             return 1
        minC = float('inf')
        for coin in self.coins:
            minVal = self.rec(rem - coin)
            if(minVal != -1):
                minC = min(minC,minVal)
        if(minC == float('inf')):
            return -1
        return minC + 1
                
solution = Solution()
#res = solution.coinChange([1,3,5],11)
#res = solution.coinChange([5],1)
#res = solution.coinChange([411,412,413,414,415,416,417,418,419,420,421,422],9864)
res = solution.coinChange([71,440,63,321,461,310,467,456,361],9298)
print(res)

