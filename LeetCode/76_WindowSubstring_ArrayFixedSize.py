import collections

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t: 
            return ""
        
        #Add t to Dict
        dt = collections.Counter(t)
       
        #add left        
        arr = [0 for i in range(0,len(s))]
        idxL = 0

        #Find all index where number exists
        for right in range(0,len(s)): #O(s)
            if(s[right] in dt):
                arr[idxL] = right
                idxL+=1
        
        maxI = idxL-1
        #for right in range(idxL,len(s)):
        #    arr[right] = -1 #fill remaining

        #Now search
        count = 0
        result = float('inf'),0,0
        ndt={}
        left=right=arr[0]
        idxL=0
        idxR=0
        while(idxR <= maxI):
            while(count < len(t) and idxR <= maxI):
                right = arr[idxR]
                ndt[s[right]]= ndt.get(s[right],0) + 1
                if(ndt[s[right]] <= dt[s[right]]):
                    count +=1
                idxR += 1
                
                
            while(idxL <= idxR and count == len(t)):      
                left = arr[idxL]
                if right - left + 1  < result[0]:    
                    result = right - left + 1,left,right
                ndt[s[left]] -= 1
                if(ndt[s[left]] < dt[s[left]]): 
                    count -= 1
                idxL += 1

        return  "" if result[0] == float("inf") else s[result[1]:result[2]+1]
    
    
s = Solution()
res = s.minWindow("aaaaaaaaaaaabbbbbcdd","abcdd")
#res = s.minWindow("acb","bc")
#res = s.minWindow("a","aa")
#res = s.minWindow("a","a")
#res = s.minWindow("abdc","bc")
res = s.minWindow("aabdc","abc")
#res = s.minWindow("ADOBECODEBANC","ABC")
print("Result",res)