from typing import Dict, List

class Node:
    def __init__(self):
        self.incoming = 0
        self.outgoing = []

#T: O(N)
#S: O(N)
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        cdict = {}
        
        #O(E): E = number of edges
        for pair in prerequisites:
            post,pre = pair[0],pair[1]
            if pre not in cdict:
                cdict[pre] = Node()
            cdict[pre].incoming += 1
            if post not in cdict:
                cdict[post] = Node()
            cdict[post].outgoing.append(pre)
           

        q = []
        order = []
        #O(N) - Number of courses/Nodes
        for i in range(numCourses):
            if(i not in cdict): #Has no dependencies at all
                order.append(i) 
            else:
                if(cdict[i].incoming == 0):
                    q.append(i)
        
        
        #[0,1], [1,2]
        #cdict[1].incoming = 1, cdict[0].outgoing=[1]
        #cdict[2].incoming = 1, cdict[1].outgoing = [2] 
        #Topological Sort
        noEdges = 0
        #O(E)
        while len(q) > 0:
            i = q.pop(0)
            order.append(i)
            for id in cdict[i].outgoing:
                noEdges += 1
                cdict[id].incoming -= 1
                if(cdict[id].incoming==0):
                    q.append(id)
        

        if(noEdges < len(prerequisites)):
            return []
        order.reverse()
        return order

solution = Solution()

#preReq = [[0,1],[1,2]] #2,1,0
#numCourses = 3
preReq = [[0,2],[1,2],[2,0]] #4,1,2,3 or 4,2,1,3
numCourses = 3
#preReq = [[1,4],[2,4],[3,1],[3,2]] #4,1,2,3 or 4,2,1,3
#numCourses = 5

val = solution.findOrder(numCourses,preReq)
print(val)

            