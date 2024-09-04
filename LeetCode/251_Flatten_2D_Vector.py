from typing import Dict, List, Optional

class Vector2D:
    #S: O(1)
    def __init__(self, vec: List[List[int]]):
        self.il = 0 #list
        self.isl = 0 #sub-list
        self.vec = vec
        while(self.il < len(self.vec) and len(self.vec[self.il])==0): #T:O(n)
            self.il += 1

    def next(self) -> int:
        val = self.vec[self.il][self.isl]
        if(self.isl == len(self.vec[self.il])-1): #was last element
            self.il += 1
            self.isl = 0
            while(self.il < len(self.vec) and len(self.vec[self.il])==0): #T:O(n)
                self.il += 1
        else:
            self.isl += 1
        return val

    def hasNext(self) -> bool:
        if(self.il < len(self.vec)):
            return True
        else:
            return False
        

# Your Vector2D object will be instantiated and called as such:
vec = [[1, 2], [3], [4]]
obj = Vector2D(vec)
param_1 = obj.next()
param_1 = obj.next()
param_1 = obj.next()
param_2 = obj.hasNext()
param_2 = obj.hasNext()
param_1 = obj.next()
param_2 = obj.hasNext()
print(param_1,param_2)