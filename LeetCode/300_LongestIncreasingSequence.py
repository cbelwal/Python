from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        self.dp = [1] * len(nums) #Index i shows max subsquence length till this index.

        #      10,9,2,5,3,7,101,18: len = 8
        # dp = 1 ,1,1,2,2,3,1  ,1
        #T :O(n^2)
        #S: O(n)
        for i in range(1,len(nums)):
            for j in range(0,i):
                if(nums[j] < nums[i]):
                    self.dp[i] = max(self.dp[i],self.dp[j]+1)

        return max(self.dp)

        
    
solution = Solution()
#res = solution.lengthOfLIS([10,9,2,5,3,7,101,18])
res = solution.lengthOfLIS([0,1,0,3,2,3])
print("Result:",res)

    
