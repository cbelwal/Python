from typing import Dict, List
import heapq

class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        results = []
        edges = []
        
        for i,blg in enumerate(buildings): #assume in sorted order of x
            edges.append((blg[0],i))
            edges.append((blg[1],i))

        edges.sort() #in sorted order
        pq = []
        idx = 0

        while idx < len(edges):
            cx = edges[idx][0]

            while idx < len(edges) and edges[idx][0]==cx:
                i = edges[idx][1]

                #Add height to pq
                right = buildings[i][1] 
                h = buildings[i][2]
                heapq.heappush(pq,(-h,right))

                while pq and pq[0][1] <= cx:
                    heapq.heappop(pq)
                idx += 1
            
            maxH = - pq[0][0] if pq else 0

            if not results or maxH != results[-1][1]:
                results.append([cx,maxH])

        return results

solution = Solution()
val = solution.getSkyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]])
#val = solution.getSkyline([[0,2,3],[2,5,3]])
print(val)