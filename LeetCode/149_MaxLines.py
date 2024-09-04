from typing import List

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        maxCount = 0
        cM =0
        cC = 0
        if(len(points) <= 2):
            return len(points)
        for i in range(0,len(points)-1):
            dictM = {}
            for j in range(i+1,len(points)):
                cM = self.getM(points[i],points[j])
                if(cM in dictM):
                    dictM[cM] += 1
                    maxCount = max(dictM[cM],maxCount)
                else:
                    dictM[cM] = 2
                    maxCount = max(dictM[cM],maxCount) 
        return maxCount
        
   
    def getM(self,p1:List[int],p2:List[int])->float:
        if(p2[0] != p1[0]):
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        else:
            m = 0
        return m
    
    def getC(self,p1:List[int],m:float)->float:
        #y = mx+C => y - mx = C
        c = p1[1] - m*p1[0]
        return c

solution = Solution()
#val = solution.maxPoints([[1,1],[2,2],[3,3]])
#val = solution.maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]])
val = solution.maxPoints([[-6,-1],[3,1],[12,3]])

print(val)