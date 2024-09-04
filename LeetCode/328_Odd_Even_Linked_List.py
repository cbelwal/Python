from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        store = [0] * 10001
        idx=0
        node=head
        while(node != None):
            store[idx] = node.val
            idx += 1
            node = node.next
        
        # Do odd assignments
        node=head
        for i in range(0,idx,2):
            node.val = store[i]
            node = node.next

        # Do even assignments
        for i in range(1,idx,2):
            node.val = store[i]
            node = node.next
        
        return head

def convertToLinkedList(items):
    head = ListNode()
    node = head
    for item in items:
        node.val = item
        node.next = ListNode()
        node = node.next
    return head

def printLinkedList(head):
    node = head
    while(node.next != None):
        print(node.val)
        node= node.next

solution = Solution()

head = solution.oddEvenList(convertToLinkedList([1,2,3,4,5]))
#head = solution.oddEvenList(convertToLinkedList([2,1,3,5,6,4,7]))
printLinkedList(head)
