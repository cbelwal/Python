import math

class Solution:
    def countPrimes(self, n: int) -> int:
        composites = {}
        
        if(n <= 2):
            return 0
        
        s = int(math.sqrt(n))
        for i in range(2,s+1):
            start = i*i
            if(start not in composites):
                for j in range(start,n,i):
                    composites[j] = 1    

        return (n - len(composites) - 2) # return number of primes excluding 1 and number

solution = Solution()
val = solution.countPrimes(10)
#val = solution.countPrimes(499979)
print(val)