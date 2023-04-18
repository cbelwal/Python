class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:        
        def decimalToString(dividend:float, divisor:float)->str:
            smap = {}
            idx = 0
            s = ""

            while (dividend != 0):
                if(dividend in smap):
                    sidx = smap[dividend]
                    s = s[0:sidx] + "(" + s[sidx:len(s)] + ")" 
                    return s

                # Resume from here.
                smap[dividend] = idx
                dividend = dividend * 10 
                s = s + str(dividend // divisor) 
                dividend = dividend % divisor 
                idx += 1

            return s
        
        mainS = ""
        
        sign = ""
        if(numerator < 0 and denominator > 0 or
           numerator > 0 and denominator < 0):
            sign = "-"
        
        numerator = abs(numerator)
        denominator = abs(denominator)

        left = numerator//denominator
        mainS = str(left)
        
        mod = numerator % denominator
        if (mod == 0):
            return sign + mainS

        mainS = sign + mainS + "." + decimalToString(mod,denominator)

        return mainS

solution = Solution()

#val = solution.fractionToDecimal(1,2)
#val = solution.fractionToDecimal(2,1)
#val = solution.fractionToDecimal(4,333)
#val = solution.fractionToDecimal(-4,333)
#val = solution.fractionToDecimal(4,-333)
#val = solution.fractionToDecimal(-1,-4)
#val = solution.fractionToDecimal(-2147483648,-1)
val = solution.fractionToDecimal(-2147483648,1)
#val = solution.fractionToDecimal(1,6)
#val = solution.fractionToDecimal(1,333)

print(val)