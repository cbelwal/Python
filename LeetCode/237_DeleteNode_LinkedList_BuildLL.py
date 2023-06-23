# Definition for singly-linked list.
class ListNode:
     def __init__(self, x):
         self.val = x
         self.next = None

class Solution:
    #T: O(1)
    #S: O(1) No extra space
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next        
        
def buildLL(numbers):
    root = ListNode(numbers[0])
    node = root

    for i in range(1,len(numbers)):
        node.next = ListNode(numbers[i])
        node = node.next
    return root

def printLL(root):
    while(root != None):
        print(root.val)
        root = root.next

solution = Solution()
root = buildLL([4,5,1,9])
solution.deleteNode(root.next)
printLL(root)


