from typing import Optional

# Definition for a binary tree node.
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val 
         self.left = left
         self.right = right

class Solution:
     def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.dict = {}
        if(root is None): return 0
        return self.rec(0,root)
        
        
     def rec(self,cSum:int, node:TreeNode):
        sum = 0 
        maxSum = 0
        if(node.left is None and node.right is None):
            return cSum + node.val
        
        if(node.val in self.dict):
            return self.dict[node.val]
        
        tmpL = 0
        tmpR = 0
        if(node.left is not None):
            tmpL = self.rec(cSum + node.val,node.left)
        if(node.right is not None):
            tmpR = self.rec(cSum + node.val,node.right)
        maxSum = max(maxSum, max(tmpL,tmpR))

        if(node.left is not None):
            tmpL = self.rec(node.val,node.left)
        if(node.right is not None):
            tmpR = self.rec(node.val,node.right)
        maxSum = max(maxSum, max(tmpL,tmpR))
        
        tmpL = 0
        tmpR = 0
        if(node.left is not None):
            tmpL = self.rec(0,node.left)
            maxSum = max(maxSum,max(node.val, tmpL))
        if(node.right is not None):
            tmpR = self.rec(0,node.right)
            maxSum = max(maxSum,max(node.val, tmpR))
        
        #maxSum = max(maxSum, node.val + tmpL + tmpR)
        #print("MaxSum",maxSum)
        
        self.dict[node.val] = maxSum
        return maxSum

def BuildTree()->TreeNode:
    root = TreeNode(1)
    root.left = TreeNode(-2) 
    root.right = TreeNode(-3)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(3)
    root.right.left = TreeNode(-2)
    root.left.left.left = TreeNode(-1)
    return root

root = BuildTree()
solution = Solution()
ans = solution.maxPathSum(root)
print(ans)