from typing import List

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        store = [-1] * (10**5+1) # Use max Space
        for n in nums:
            if(store[n]==-1):
                store[n] = n
            else:
                return n
        return None
    