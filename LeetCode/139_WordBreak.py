from typing import Dict, List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        def checkWord(start):
            if(start == len(s)):
                return True
            
            for end in range(start+1,len(s)+1):
                if s[start:end] in wordSet:
                        if (checkWord(end)):
                            return True
            return False
            
        wordSet = set(wordDict)
        return checkWord(0)

solution = Solution()
val = solution.wordBreak("ccaccc",["cc","ac"])
#val = solution.wordBreak("leetcode",["leet","code"])
print(val)