from typing import Dict, List

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        def updateSegTree(segTree,pos,value,sizeBucket):
            pos += sizeBucket
            segTree[pos] = value
            #Now Update parents
            while(pos >  1):
                pos = pos // 2
                segTree[pos] = segTree[2*pos] + segTree[2*pos+1]

        def querySegTree(segTree,left,right,sizeBucket)->int:
            result = 0
            left += sizeBucket
            right += sizeBucket

            while (left < right):
                if left % 2 == 1:
                   result += segTree[left]     
                   left = left + 1 
                left = left //2
                if right % 2 == 1:
                    right = right - 1
                    result += segTree[right]
                right = right //2
            return result

        # --- Testing
        #offset = 0
        #size = 12
        #---- Prod
        offset =  10**4
        size = 2 * offset + 1
        #----
        bucket = [0] * size
        bucketSize = len(bucket)
        segTree = [0] * (2 * bucketSize)
        
        counts = [0] * len(nums)
        for i in range(len(nums)-1,-1,-1):
            counts[i] = querySegTree(segTree,0,offset + nums[i],bucketSize)
            bucket[offset+nums[i]] += 1
            updateSegTree(segTree,offset+nums[i],bucket[offset+nums[i]],bucketSize)    
            
        return counts

s = Solution()
#res = s.countSmaller([-1])
res = s.countSmaller([-1,-1])
#res = s.countSmaller([5,2,6,1])
#res = s.countSmaller([3,2,4,1])
print("Result",res)