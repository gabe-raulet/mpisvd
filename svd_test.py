import sys
import numpy as np
import subprocess as sp
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

def extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, m, n, q, p):
    b, s = param_check_and_get(m, n, q, p)
    assert Aq1_11.shape[0] == m and Aq1_11.shape[1] == p
    assert Aq1_12.shape[0] == m and Aq1_12.shape[1] == p
    assert Vtq1_11.shape[0] == p and Vtq1_11.shape[1] == (2**(q-1))*s
    assert Vtq1_12.shape[0] == p and Vtq1_12.shape[1] == (2**(q-1))*s
    Aq1 = np.concatenate((Aq1_11, Aq1_12), axis=1)
    Vhtq11 = np.concatenate((Vtq1_11, np.zeros(Vtq1_11.shape)), axis=1)
    Vhtq12 = np.concatenate((np.zeros(Vtq1_12.shape), Vtq1_12), axis=1)
    Vhtq1 = np.concatenate((Vhtq11, Vhtq12), axis=0)
    Uq, Sq, Vtq = svds(Aq1, p)
    W = (Vtq@Vhtq1).T
    Qq, Rq = np.linalg.qr(W)
    Aq = Uq@np.diag(Sq)@np.linalg.inv(Rq)
    Uc, Sc, Vtp = svds(Aq, p)
    Vtc = Vtp@Qq.T
    return Uc, Sc, Vtc

def binary_comb(A, p, q):

    m, n = A.shape
    b, s = param_check_and_get(m, n, q, p)
    Adict = {}
    Vtdict = {}

    for i in range(b):
        A1_i, Vt1_i = seed_node(A, i, q, p)
        Adict[i] = A1_i
        Vtdict[i] = Vt1_i

    for k in range(1, q):
        for i in range(2**(q-k)):
            Ak_2i_0, Ak_2i_1 = Adict[2*i], Adict[2*i+1]
            Vtk_2i_0, Vtk_2i_1 = Vtdict[2*i], Vtdict[2*i+1]
            Ak1_lj, Vtk1_lj = combine_node(Ak_2i_0, Vtk_2i_0, Ak_2i_1, Vtk_2i_1, m, n, i, k, q, p)
            Adict[i] = Ak1_lj
            Vtdict[i] = Vtk1_lj

    Aq1_11 = Adict[0]
    Aq1_12 = Adict[1]
    Vtq1_11 = Vtdict[0]
    Vtq1_12 = Vtdict[1]

    mmwrite("Aq1_11_c.mtx", Aq1_11);
    mmwrite("Aq1_12_c.mtx", Aq1_12);
    mmwrite("Vtq1_11_c.mtx", Vtq1_11);
    mmwrite("Vtq1_12_c.mtx", Vtq1_12);

    Up, Sp, Vtp = extract_node(Aq1_11, Vtq1_11, Aq1_12, Vtq1_12, m, n, q, p)

    return Up, Sp, Vtp, Adict, Vtdict

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

def check_errors(A, U, S, Vt, Urand, Srand, Vtrand, p):
    assert Urand.shape[0] == A.shape[0] and Urand.shape[1] == p and len(Srand) == p and Vtrand.shape[0] == p and Vtrand.shape[1] == A.shape[1]
    Up, Sp, Vtp = U[:,:p], S[:p], Vt[:p,:]
    Arand = Urand@np.diag(Srand)@Vtrand
    Aerr = np.linalg.norm(A-Arand)
    Uerr = np.linalg.norm(Up@Up.T - Urand@Urand.T)
    Serr = np.linalg.norm(Sp - Srand)
    Vterr = np.linalg.norm(Vtp.T@Vtp - Vtrand.T@Vtrand)
    print(f"Aerr={Aerr:.18e}")
    print(f"Uerr={Uerr:.18e}")
    print(f"Serr={Serr:.18e}")
    print(f"Vterr={Vterr:.18e}\n")

cond = 100.
damp = 2.

# scalers = [1, 4, 16]
# scalers = [4, 16]
scalers=[16]

for n in [128, 256, 1024]:
    for scale in scalers:
        m = scale * n
        r = n
        p = 10
        for q in [3]:
            s = n / (2**q)
            if p > s: continue

            A, U, S, Vt = create_example(m, n, r, p, q, cond, damp)
            mmwrite("A.mtx", A)

            Urand, Srand, Vtrand, Adict, Vtdict = binary_comb(A, p, q)
            check_errors(A, U, S, Vt, Urand, Srand, Vtrand, p)

            proc = sp.Popen(["./full_svd", "A.mtx", str(p), str(q)], stdout=sp.PIPE)
            proc.wait()

            if proc.returncode != 0:
                print(f"./full_svd exited with non-zero status")
                sys.exit(1)

            Scheck = np.diag(mmread("Sp_a.mtx"))
            Ucheck = mmread("Up_a.mtx")
            Vtcheck = mmread("Vtp_a.mtx")

            check_errors(A, U, S, Vt, Ucheck, Scheck, Vtcheck, p)

