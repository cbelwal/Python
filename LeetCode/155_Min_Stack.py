class Node:
    left=None
    right=None
    value = 0
    minimum = 0

class MinStack:
    def __init__(self):
        self.head = Node()
        self.min = 2**31

    def push(self, val: int) -> None:
        node = Node()
        node.value = val
        node.right = self.head
        node.right.left = node
        node.minimum = self.min
        self.head = node
        if(val < self.min):
            self.min = val

    def pop(self) -> None:
        self.head = self.head.right
        self.min = self.head

    def top(self) -> int:
        return self.head.value

    def getMin(self) -> int:
        return self.min


# Your MinStack object will be instantiated and called as such:
val = 3
obj = MinStack()
obj.push(val)
#obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()
print(param_4)