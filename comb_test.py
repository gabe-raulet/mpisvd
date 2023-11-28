import sys
import numpy as np
from scipy.io import mmread, mmwrite

def ispow2(v):
    if v == 0: return False
    return not bool(v&(v-1))

def param_check_and_get(m, n, q, p):
    b = 2**q
    s = n//b
    assert ispow2(m)
    assert ispow2(n)
    assert q >= 1
    assert n >= b
    assert s >= p and p >= 1
    return b, s

def _svds(A, p):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:,:p], S[:p], Vt[:p,:]

def svds(A, p):
    m, n = A.shape
    assert p <= min(m,n)
    if m >= n:
        return _svds(A, p)
    else:
        Uz, Sz, Vtz = _svds(A.T, p)
        return Vtz.T, Sz, Uz.T

def combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, i, k, q, p):
    b, s = param_check_and_get(m, n, q, p)
    c = 2**(q-k) # number of nodes at level k of binary tree
    d = (2**(k-1)) * s
    assert 0 <= i < c
    assert 1 <= k < q
    assert Ak_2i_0.shape[0] == m and Ak_2i_0.shape[1] == p
    assert Ak_2i_1.shape[0] == m and Ak_2i_1.shape[1] == p
    assert Vtk_2i_0.shape[0] == p and Vtk_2i_0.shape[1] == d
    assert Vtk_2i_1.shape[0] == p and Vtk_2i_1.shape[1] == d
    Aki = np.concatenate((Ak_2i_0, Ak_2i_1), axis=1)
    Vhtki1 = np.concatenate((Vtk_2i_0, np.zeros(Vtk_2i_0.shape)), axis=1)
    Vhtki2 = np.concatenate((np.zeros(Vtk_2i_1.shape), Vtk_2i_1), axis=1)
    Vhtki = np.concatenate((Vhtki1, Vhtki2), axis=0)
    Uki, Ski, Vtki = svds(Aki, p)
    W = (Vtki@Vhtki).T
    Qki, Rki = np.linalg.qr(W)
    Ak1_lj = Uki@np.diag(Ski)@np.linalg.inv(Rki)
    Vtk1_lj = Qki.T
    return Ak1_lj, Vtk1_lj

def create_example(m, n, r, p, q, cond, damp):
    mz = max(m, n)
    nz = min(m, n)
    Az, Uz, Sz, Vtz = _create_example(mz, nz, r, p, q, cond, damp)

    if m < n:
        return Az.T, Vtz.T, Sz, Uz.T
    else:
        return Az, Uz, Sz, Vtz

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.stderr.write(f"usage: {sys.argv[0]} A.mtx Aseed.mtx Vtseed.mtx p q")
        sys.stderr.flush()
        sys.exit(1)

    A = mmread(sys.argv[1])
    Aseed = mmread(sys.argv[2])
    Vtseed = mmread(sys.argv[3])
    p = int(sys.argv[4])
    q = int(sys.argv[5])

    m, n = A.shape
    b, s = param_check_and_get(m, n, q, p)

    Ak_2i_0 = Aseed[:,0:p]
    Ak_2i_1 = Aseed[:,p:2*p]
    Vtk_2i_0 = Vtseed[:,0:s]
    Vtk_2i_1 = Vtseed[:,s:2*s]

    Ak1_lj, Vtk1_lj = combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, 0, 1, q, p)

    mmwrite("Ak1_lj.mtx", Ak1_lj)
    mmwrite("Vtk1_lj.mtx", Vtk1_lj)


