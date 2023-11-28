import sys
import numpy as np
from scipy.io import mmread, mmwrite

def ispow2(v):
    if v == 0: return False
    return not bool(v&(v-1))

def param_check_and_get(m, n, q, p):
    b = 2**q
    s = n//b
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

def seed_node(A, i, q, p):
    m, n = A.shape
    b, s = param_check_and_get(m, n, q, p)
    assert 0 <= i and i < b
    Ai = A[:, i*s:(i+1)*s]
    Ui, Si, Vti = svds(Ai, p)
    return Ui@np.diag(Si), Vti


def combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, i, k, q, p):
    b, s = param_check_and_get(m, n, q, p)
    c = 2**(q-k) # number of nodes at level k of binary tree
    d = (2**(k-1)) * s
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

def _create_example(m, n, r, p, q, cond, damp):
    param_check_and_get(m, n, q, p)
    assert 1 <= r <= n

    A = np.random.random((m,n))
    S = np.zeros(n, dtype=np.double)

    S[0] = cond
    for i in range(1, r):
        S[i] = S[i-1]/damp

    U = np.linalg.qr(np.random.random((m,n)))[0]
    Vt = np.linalg.qr(np.random.random((n,n)))[0]

    A = U@np.diag(S)@Vt

    return A, U, S, Vt


def create_example(m, n, r, p, q, cond, damp):
    mz = max(m, n)
    nz = min(m, n)
    Az, Uz, Sz, Vtz = _create_example(mz, nz, r, p, q, cond, damp)

    if m < n:
        return Az.T, Vtz.T, Sz, Uz.T
    else:
        return Az, Uz, Sz, Vtz

if __name__ == "__main__":

    m = 256
    n = 128
    r = 128
    p = 10
    q = 3
    cond = 100.
    damp = 2.

    A, U, S, Vt = create_example(m, n, r, p, q, cond, damp)

    b, s = param_check_and_get(m, n, q, p)
    Acat = []
    Vtcat = []

    for i in range(b):
        A1_i, Vt1_i = seed_node(A, i, q, p)
        Acat.append(A1_i)
        Vtcat.append(Vt1_i)

    Ak_2i_0 = Acat[0]
    Ak_2i_1 = Acat[1]
    Vtk_2i_0 = Vtcat[0]
    Vtk_2i_1 = Vtcat[1]

    mmwrite("Ak_2i_0.mtx", Ak_2i_0)
    mmwrite("Ak_2i_1.mtx", Ak_2i_1)
    mmwrite("Vtk_2i_0.mtx", Vtk_2i_0)
    mmwrite("Vtk_2i_1.mtx", Vtk_2i_1)

    Ak1_lj, Vtk1_lj = combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, 0, 1, q, p)

    mmwrite("A.mtx", A)
    mmwrite("Ak1_lj_correct.mtx", Ak1_lj)
    mmwrite("Vtk1_lj_correct.mtx", Vtk1_lj)


