# Definition for a binary tree node.
class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        #Implement BFS
        res = []
        q = []
        q.append(root)
        while(len(q) > 0):
            node = q.pop(0)
            if(node is None):
                res.append('null')
            else:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
        #Now create string
        result = ""
        startAdd = False
        for i in range(len(res)-1,-1,-1): 
            if(res[i] != 'null'):
                startAdd = True
            if(startAdd):
                result = res[i] + "," + result
            
        result = result[:-1]
        return result

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        allN = data.split(",")
        root = TreeNode(allN[0])
        
        q = []
        q.append(root)
        i = 1
        while(len(q) > 0):
            if i >= len(allN) :
                break
            node = q.pop(0)
            if(node == None):
                continue
            if allN[i] != 'null':
                node.left = TreeNode(allN[i])
                q.append(node.left)
            else:
                q.append(None)
            i += 1
            
            if allN[i] != 'null':
                node.right = TreeNode(allN[i])
                q.append(node.right)
            else:
                q.append(None)
            i += 1  
        return root

solution = Codec()
root = solution.deserialize("1,2,3,null,null,4,5")
deS = solution.serialize(root)
print(deS)