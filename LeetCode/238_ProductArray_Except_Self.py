from typing import Dict, List, Optional

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        l = len (nums)

        # leftMul = [1,2,6,24]
        leftMul,rightMul,results = [0] * l,[0] * l,[0] * l 
      
        leftMul[0] = nums[0]
        rightMul[l-1] = nums[l-1]
        for i in range(1,l):
            leftMul[i] = leftMul[i-1] * nums[i]
            rightMul[l-1-i] = rightMul[l-i] * nums[l - 1 - i]
        
        #rightMul[l-1] = nums[l-1]
        #for i in range(l-2,1,-1):
        #    leftMul[i] = leftMul[i+1] * nums[i]
        
        for i in range(0,l):
            if(i==0):
                results[0] = rightMul[i+1]
            elif(i==l-1):
                results[i] = leftMul[i-1] 
            else:
                results[i] = leftMul[i-1] * rightMul[i+1]
        
        return results
        # leftMul = [1,2,6,24] 
        # rightMul = [24,24,12,4]
        #result = leftMul[i-1] * rightMul[i+1]

solution = Solution()
val = solution.productExceptSelf([1,2,3,4])
print(val)
