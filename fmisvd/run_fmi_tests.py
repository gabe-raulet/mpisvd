import sys
import numpy as np
import subprocess as sp
from pathlib import Path
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

def create_test_parameters(colexps=[8,9,10], scalexps=[0,2,4], ranktests=4, pvals=[5,10,20], qvals=[3,5,7], cond=100., damp=2.):
    colvals = [2**k for k in colexps] # 256, 512, 1024
    scalers = [2**k for k in scalexps] # 1, 4, 16
    mnpairs = [(s*n, n) for s in scalers for n in colvals]
    tests = []
    for m,n in mnpairs:
        rs = [n//(2**i) for i in range(ranktests) if n % (2**i) == 0]
        pqs = [(p, q) for p in pvals for q in qvals if p <= n and (n/2**q) >= p]
        for r in rs:
            for p, q in pqs:
                tests.append((m, n, p, q, r, cond, damp))
    return tests

def read_diag(fname):
    nums = [float(line.rstrip()) for line in open(fname, "r")]
    return np.array(nums, dtype=np.double)

def run_test(test_params, cnt):
    m, n, p, q, r, cond, damp = test_params
    sys.stderr.write(f"test({cnt})[m={m}, n={n}, p={p}, b={2**q}, cond={cond}, damp={damp}]\n")
    sys.stderr.flush()

    toks = [m,n,p,r,cond,damp]

    sys.stderr.write(" ".join(["./build/fmi_svd", f"test{cnt}2"] + [str(tok) for tok in toks]) + "\n")
    sys.stderr.flush()

    procs = [None] * (1<<q)

    for rank in range(1<<q):
        cmd = ["./build/fmi_svd", str(rank), str(1<<q), f"test{cnt}2"] + [str(tok) for tok in toks]
        procs[rank] = sp.Popen(cmd, stdout=sp.PIPE)
    
    for rank in range(1<<q):
        procs[rank].wait()
        if procs[rank].returncode != 0:
            sys.stderr.write(f"./build/fmi_svd exited with non-zero status on test({cnt})")
            sys.stdout.flush()
            return

    A = mmread(f"A_test{cnt}{2}.mtx")
    Up = mmread(f"Up_test{cnt}{2}.mtx")
    Sp = read_diag(f"Sp_test{cnt}{2}.diag")
    Vtp = mmread(f"Vtp_test{cnt}{2}.mtx")

    Ap = Up@np.diag(Sp)@Vtp

    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    U = U[:,:p]
    S = S[:p]
    Vt = Vt[:p,:]

    Aerr = np.linalg.norm(A - Ap)
    Uerr = np.linalg.norm(U@U.T - Up@Up.T)
    Serr = np.linalg.norm(S[:p] - Sp)
    Vterr = np.linalg.norm(Vt.T@Vt - Vtp.T@Vtp)

    sys.stderr.write(f"test({cnt})[Aerr={Aerr:.8e}, Serr={Serr:.8e}, Uerr={Uerr:.8e}, Vterr={Vterr:.8e}]\n")
    sys.stderr.flush()

if __name__  == "__main__":

    # tests = create_test_parameters(colexps=[7,8,9], scalexps=[1], ranktests=1, pvals=[10], qvals=[3])
    tests = create_test_parameters(colexps=[8,10], scalexps=[0,1], ranktests=2, pvals=[10], qvals=[3,5])
    # tests = create_test_parameters(colexps=[8], scalexps=[0], ranktests=2, pvals=[10], qvals=[3,4])

    for cnt, test in enumerate(tests):
        run_test(test, cnt)
