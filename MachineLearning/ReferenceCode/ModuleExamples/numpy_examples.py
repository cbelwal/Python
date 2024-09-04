import numpy as np

mat1 = np.random.randn(3, 2)
mat2 = np.random.randn(2, 3)

print("Result of matrix multiplication:", np.matmul (mat1,mat2) )

val1 = [1,2,3,4]
val2 = [1,2,3,4]

print("Covariance",np.cov(val1,val2))
