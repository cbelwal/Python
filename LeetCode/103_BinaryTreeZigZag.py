# Definition for a binary tree node.
from typing import Optional, List

class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        results = []
        if(root is None):
            return results
        clevel = 0
        q = [(root,clevel)]
        tmp = []
        while(len(q) > 0):
            node,level = q.pop(0)
            if(node != None):
                if(level > clevel):
                    if(level % 2 == 0): #previous level
                        #reverse
                        tmp.reverse()
                        results.append(tmp)
                    else:
                        results.append(tmp)
                    clevel += 1
                    tmp = []
                
                q.append((node.left,level+1))
                q.append((node.right,level + 1))
               
                tmp.append(node.val)
            
        results.append(tmp)
        return results

def BuildTree()->TreeNode:
    root = TreeNode(3)
    root.left = TreeNode(9) 
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    return root

root = BuildTree()
solution = Solution()
ans = solution.zigzagLevelOrder(root)
print(ans)