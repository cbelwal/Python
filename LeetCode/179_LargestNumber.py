from typing import Dict, List

class LargerNumKey(str):
    def __lt__(x, y):
        return x+y > y+x
    
class Solution:    
    def largestNumber(self, nums: List[int]) -> str:
        snums = [str(i) for i in nums]

        snums = sorted(snums,key=LargerNumKey)
        
        result = ""
        for snum in snums:
            result = result + snum

        return result 

solution = Solution()
val = solution.largestNumber([3,30,34,5,9])
print(val)