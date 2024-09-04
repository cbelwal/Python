from typing import Dict, List

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n
        
        start=count=0
        #S: O(1)
        while count < n: #O(n) Loop will repeat n times
            idx,prev = start,nums[start]
            while (True):
                nums[(idx+k)%n],prev = prev,nums[(idx+k)%n]
                count += 1
                idx = (idx + k)%n
                if(idx == start):
                    break
            start += 1

solution = Solution()
#array = [1,2,3,4,5,6,7]
array = [1,2]
solution.rotate(array,2)
print(array)