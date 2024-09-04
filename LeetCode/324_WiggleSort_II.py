from typing import List

class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        #1,5,1,1,6,4
        #1,1,1,1,4,5,6, half = 3
        #1,4,1,1,5,6
        store = [0] * 5000
        maxNum = 0
        for num in nums:
            store[num] += 1
            if(num > maxNum):
                maxNum = num

        idx = 1
        l = len(nums)
        for i in range(maxNum,-1,-1):
            while(store[i] != 0 and idx<l):
                nums[idx] = i
                store[i] -= 1
                idx += 2

        idx = 0
        l = len(nums)
        for i in range(maxNum,-1,-1):
            while(store[i] != 0 and idx<l):
                nums[idx] = i
                store[i] -= 1
                idx += 2
            if(idx >= l):
                break

solution = Solution()
nums = [1,5,1,1,6,4]
solution.wiggleSort(nums)
print(nums)  

        