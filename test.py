import numpy as np
from scipy.io import mmread

A = mmread("A.mtx")
S = np.diag(mmread("S.mtx"))

Sc = np.linalg.svd(A, compute_uv=False)

print(f"{np.linalg.norm(S-Sc)}:.18e")
