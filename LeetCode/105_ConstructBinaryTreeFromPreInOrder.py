# Definition for a binary tree node.
from typing import Optional, List

class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

class Solution:
        def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
            #inorder: left -> parent -> Right
            #Preorder: parent -> left -> Right 
            #Store Inorder in dict
            inDict = {}
            for i in range(0,len(inorder)):
                inDict[inorder[i]] = i

            self.preIdx = 0
            #Pre: 3,9,20,15,7
            #In : 9,3,15,20,7

            def BuildTree(inStartIdx,inEndIdx)->TreeNode: #0,0
                if(inStartIdx < 0 or inEndIdx >= len(inorder) or
                        inStartIdx > inEndIdx):
                    return None
                
                node = TreeNode(preorder[self.preIdx] )
                splitIdx = inDict[node.val] #1
                self.preIdx += 1 #1

                if(inStartIdx != inEndIdx):
                    node.left =  BuildTree(inStartIdx,splitIdx-1)
                    node.right =  BuildTree(splitIdx+1,inEndIdx)
                
                return node

            root = BuildTree(0,len(inorder)-1)            
            return root

solution = Solution()
#inorder = [1,4,2,5,3,6,7]
#preorder = [5,4,1,2,6,3,7]
inorder = [1,2]
preorder = [2,1]
ans = solution.buildTree(preorder,inorder)
print(ans)