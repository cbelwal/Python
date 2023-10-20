from typing import Optional

class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution:
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        # 1 2 3 4 5 6
        self.stack = []
        self.inOrder(root)
        #T: O(N)
        #M: O(N)
        for i in range(0,len(self.stack)):
            if (self.stack[i] == p):
                if(i < len(self.stack)-1):
                    return self.stack[i+1]
        return None        

    # T:O(N)
    # M:O(N)
    def inOrder(self,node):
        if(node.left != None):
            self.inOrder(node.left)
        self.stack.append(node)
        if(node.right != None):
            self.inOrder(node.right)