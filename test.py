import numpy as np
from scipy.io import mmread

# A = mmread("A.mtx")
# S = np.diag(mmread("S.mtx"))

# Sc = np.linalg.svd(A, compute_uv=False)

# print(f"{np.linalg.norm(S-Sc)}:.18e")

U = mmread("U.mtx")
Up = mmread("Up.mtx")
p = Up.shape[1]

U = U[:,:p]

M = U@U.T - Up@Up.T
print(f"{np.linalg.norm(M):.18e}")
