# Allocate 1D array
size = 10
array = [0] * size
print(array)

# Allocate 2D Array
rows, cols = 8, 5
matrix = [[0 for x in range(cols)] for y in range(rows)] 
print("------")
print("Matrix:",matrix)

# Implement Stack, LIFO 
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
print("------")
print("Stack Peek:",stack[-1])
print("Stack Pop:",stack.pop())
print("Remaining Stack:",stack)


# Implement Queue, FIFO
queue = []
queue.append(1)
queue.append(2)
queue.append(3)
print("------")
print("Queue Peek:",queue[0])
print("Queue Pop:",queue.pop(0))
print("Remaining Queue:",queue)

# Special function
# bisect_left(list, num, beg, end) :- This function returns the insertion position 
# in the sorted list, where the number passed in argument 
# can be placed so as to maintain the resultant list in sorted order.
import bisect
list = [5,4,7,2] #Sorted: 2,4,5,7
i = bisect.bisect_left(list, 1)  # i = 0
i = bisect.bisect_left(list, 10) # i = 4
i = bisect.bisect_left(list, 6) # i = 2 
print("Index:",i)

 