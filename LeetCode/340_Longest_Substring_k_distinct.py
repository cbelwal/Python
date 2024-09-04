class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        l = len(s)
        
        if(k==0):
            return 0
        if(l <= k):
            return l
    
        left=0
        right=left + 1
        maxFinal = float('-inf')
        dictV={}
        dictV[s[left]] = 1
        while(right < l): #O(n)
            # e c e b a
            # e b a
            if(s[right] not in dictV):
                dictV[s[right]] = 1    
            else:
                dictV[s[right]] += 1
            
            if(len(dictV) > k):
                if((right - left) > maxFinal):
                    maxFinal = right - left
                #Keep removing from left
                
                dictV[s[left]] -= 1
                if(dictV[s[left]] == 0):
                    del dictV[s[left]]
                left += 1
                    
            right += 1

        if(right - left > maxFinal):
            maxFinal = right - left

        return maxFinal
 
solution = Solution()
#res = solution.lengthOfLongestSubstringKDistinct("eceba",2)
#res = solution.lengthOfLongestSubstringKDistinct("aa",2)
#res = solution.lengthOfLongestSubstringKDistinct("a",2)
#res = solution.lengthOfLongestSubstringKDistinct("a",1)
#res = solution.lengthOfLongestSubstringKDistinct("aa",0)
#res = solution.lengthOfLongestSubstringKDistinct("aac",2)
#res = solution.lengthOfLongestSubstringKDistinct("aba",1)
res = solution.lengthOfLongestSubstringKDistinct("bacc",2)
print("Result",res)
