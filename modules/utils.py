#PISTA: es esta la mejor forma de hacer una matmul?
"""
def matmul_biasses(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    for i in range(m):
        for k in range(p):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
            #C[i][j] += bias[j]
    C+= bias
    return C
"""
def matmul_biasses(A,B,C,bias):
    C = A @ B + bias
    return C
