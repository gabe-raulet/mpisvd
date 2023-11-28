
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

def seed_node(A, i, q, p):
    m, n = A.shape
    b, s = param_check_and_get(m, n, q, p)
    assert 0 <= i and i < b
    Ai = A[:, i*s:(i+1)*s]
    Ui, Si, Vti = svds(Ai, p)
    return Ui@np.diag(Si), Vti

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
    if len(sys.argv) != 8:
        sys.stderr.write(f"usage: {sys.argv[0]} m n r p q cond damp\n")
        sys.stderr.flush()
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    r = int(sys.argv[3])
    p = int(sys.argv[4])
    q = int(sys.argv[5])
    cond = float(sys.argv[6])
    damp = float(sys.argv[7])

    A, U, S, Vt = create_example(m, n, r, p, q, cond, damp)

    m, n = A.shape
    b, s = param_check_and_get(m, n, q, p)
    Acat = []
    Vtcat = []

    for i in range(b):
        A1_i, Vt1_i = seed_node(A, i, q, p)
        Acat.append(A1_i)
        Vtcat.append(Vt1_i)

    Aseed = np.concatenate(Acat, axis=1)
    Vtseed = np.concatenate(Vtcat, axis=1)

    mmwrite("A.mtx", A)
    mmwrite("Aseed.mtx", Aseed)
    mmwrite("Vtseed.mtx", Vtseed)
