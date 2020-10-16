
import itertools
import numpy as np

def warn(condition, string):
    if condition == False:
        print("WARNING: "+string)

def logsum_exp(y):
    m = y.max()
    return m + np.log((np.exp(y - m)).sum())

# Normalizes the rows of a matrix M to sum up to 1
def normalize_rows(M):
    row_sums = M.sum(axis=1)
    return M/row_sums[:, np.newaxis]
	
# Normalizes the columns of a matrix M to sum up to 1
def normalize_columns(M):
    col_sums = M.sum(axis=0)
    return M/col_sums[np.newaxis, :]

# Calculates the maximum difference between entries of two matrices A and B
def max_diff(A, B):
    C = A-B
    return max([abs(x) for x in list(C.flatten())])

# Calculates the L1 difference between two matrices A and B
def L1_diff(A,B):
    C = A-B
    return sum([abs(x) for x in list(C.flatten())])


# Calculates lower bound on L1 error between A and B
def min_error(A, B):
    K = A[0,:].size
    if K != B[0, :].size:
        print("Matrices have different numbers of columns")
    total_err = 0
    for colA in range(K):
        min_err = float("inf")
        for colB in range(K):
            err = (abs(A[:, colA] - B[:, colB])).sum()
            if err < min_err:
                min_err = err
        total_err = total_err + min_err
    return total_err
    
# Calculates a greedy L1 error between A and B
def greedy_error(A, B):
    K = A[0,:].size
    if K != B[0, :].size:
        print("Matrices have different numbers of columns")
    total_err = 0
    columns_B = list(range(K))
    for colA in range(K):
        min_err = float("inf")
        col_index = -1
        for colB in columns_B:
            err = (abs(A[:, colA] - B[:, colB])).sum()
            if err < min_err:
                min_err = err
                col_index = colB
        total_err = total_err + min_err
        columns_B.remove(col_index)
    return total_err
