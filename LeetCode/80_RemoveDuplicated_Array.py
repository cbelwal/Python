from typing import Dict, List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # nums: 1 1  1 2  2 4 6
        # new: 1 1 2 2 4 6
        self._nums = nums
        count = 1
        orig = self._nums[0]
        k = len(self._nums)
        startNeg = 0
        countForK = 0 
        # [1,1,1,2,2,3]
        for i in range(1,len(self._nums)): # O(n)
            if self._nums[i] == orig:
                count += 1
            else:
                count = 1
                orig = self._nums[i]
        
            if(count > 2):
                if(startNeg==0): #start = 2
                    startNeg = i
                self._nums[i] = -1 # [1,1,-1,2,2,3] 
                countForK+=1
        
        # Now remove elements
        for i in range(startNeg,len(self._nums)-1): # O(n)
            count = 0
            start = i #2
            while (i < len(self._nums) and self._nums[i] == -1):
                i += 1 #3
            end = i-1
            count = end - start + 1
            for j in range(start,end+1): #j=2
                if(j+count > len(self._nums) - 1):
                    break
                self._nums[j] = self._nums[j+count]
                self._nums[j+count] = -1 #[1,1,2,-1]
            

        return len(self._nums) - countForK 
             
    
solution = Solution()
#nums = [1,1,1,2,2,3]
nums = [0,0,1,1,1,1,2,3,3]
val = solution.removeDuplicates(nums)
print(f"Result:{val},{nums}")
