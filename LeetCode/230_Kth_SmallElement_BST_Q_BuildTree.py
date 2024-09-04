from typing import Dict, List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def addToQ(node):
            if(len(q) ==k):
                return #No need to go further
            if(node.left != None):
                addToQ(node.left)
            q.append(node.val)
          
            if(node.right != None):
                addToQ(node.right)
                
        q = []
        addToQ(root)
      
        for i in range(k-1):
            q.pop(0)

        return q.pop(0)

#Takes a list and builds the tree
def buildTree(node,treeList,idx)->TreeNode:
    if(idx >= len(treeList)):
        return
    node.val=treeList[idx]
    
    if(idx*2+1 < len(treeList) and treeList[idx*2+1]!=None):
        node.left = TreeNode()
        buildTree(node.left,treeList,idx*2+1)
    
    if(idx*2+2 < len(treeList) and treeList[idx*2+2] != None):
        node.right = TreeNode()
        buildTree(node.right,treeList,idx*2+2)
        

solution = Solution()
root = TreeNode() 
buildTree(root,[3,1,4,None,2],0)
k= 1
val = solution.kthSmallest(root,k)
#val = solution.getSkyline([[0,2,3],[2,5,3]])
print(val)
    
        
       