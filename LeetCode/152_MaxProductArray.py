from typing import Dict, List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        #2,6,-12,4
        maxp,minp,maxm = nums[0],nums[0],nums[0]
        for i in range (1,len(nums)):
            oldmaxp = maxp
            maxp = max(nums[i],nums[i]*maxp,nums[i]*minp)
            minp = min(nums[i],nums[i]*oldmaxp,nums[i]*minp)
            maxm = max(maxm,maxp)
        return maxm

solution = Solution()
#val = solution.maxProduct([2,3,-2,4])

#val = solution.maxProduct([2,-5,-2,-4,3])
val = solution.maxProduct([1,0,-1,2,3,-5,-2])
print(val)