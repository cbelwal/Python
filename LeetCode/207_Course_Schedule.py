from typing import Dict, List

class Node():
    def __init__(self):
        self.incoming = 0
        self.outgoing = []

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # We want to check if there are no cycles in the dependency graph
        # If there are cycles satisfying dependency is not possible
        # If there are no cycles, a topological sort order will exist
        
        nodes = {}
    
        if(len(prerequisites)==0):
            return True
        
        edges = 0
        for pr in prerequisites:
            edges += 1
            if pr[1] not in nodes:
                nodes[pr[1]] = Node()
            nodes[pr[1]].incoming += 1
            if pr[0] not in nodes:
                nodes[pr[0]] = Node()
            nodes[pr[0]].outgoing.append(pr[1])    
        
        queue = []
        for id in nodes:
            if(nodes[id].incoming == 0):
                queue.append(id)

        removedEdges = 0
        while len(queue) > 0:
            id= queue.pop(0) #Remove from queue
            node =nodes[id]

            for n in node.outgoing:
                nodes[n].incoming -= 1
                removedEdges += 1
                if(nodes[n].incoming == 0):
                    queue.append(n)
        
        if(removedEdges == edges):
            return True
        return False


solution = Solution()

preReq = [[1,4],[2,4],[3,1],[3,2]]
numCourses = 5

val = solution.canFinish(numCourses,preReq)
print(val)