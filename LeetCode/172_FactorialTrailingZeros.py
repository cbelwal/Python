class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        for i in range(5,n+1,5):
            dividend = i    
            while dividend % 5 == 0:
                count += 1
                dividend = dividend // 5 #O(1)
        return count

solution = Solution()
#res = solution.trailingZeroes(4)
#res = solution.trailingZeroes(3)
res = solution.trailingZeroes(25)
#res = solution.trailingZeroes(3704)
print(res)