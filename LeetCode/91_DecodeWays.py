class Solution:
    def numDecodings(self, s: str) -> int:
        memo = {}
        tmpS = s
        #Total cominbations
        #123 = (123),(12)3,1(23)
        #1234 = (1234),(12)(34),(12)(3)(4),1(23)4
        
        #T: O(N)
        #S: O(N) + O(N)        
        def computeGroupings(idx) -> int:
            if(idx < len(tmpS)):
                if(tmpS[idx] == '0'):
                    return 0

            if (idx >= len(tmpS)-1): #Only one char
                return 1

            if(idx in memo):
                return memo[idx] #We have been in this path

            sum = 0
            for i in range(1,3): #only 1 and 2 digits
                cIdx = int(tmpS[idx:idx+i])
                if(cIdx <= 26 and cIdx > 0):
                    sum += computeGroupings(idx+i)
                
            memo[idx] = sum
            return sum

        return computeGroupings(0)
         

solution = Solution()
#code = "11106"
#code = "12"
#code = "226"
code = "111111111111111111111111111111111111111111111"
val = solution.numDecodings(code)
print(val)     