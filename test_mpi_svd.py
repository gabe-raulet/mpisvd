import numpy as np
from scipy.io import mmread, mmwrite
from pathlib import Path

Acats = []
i = 0
while True:
    Afile = Path(f"A{i}.mtx")
    if not Afile.is_file(): break
    Acats.append(mmread(str(Afile)))
    i += 1

A = np.concatenate(Acats, axis=1)
Aref = mmread("A.mtx")
assert np.allclose(A, Aref)

U = mmread("U.mtx")
S = mmread("S.mtx")
Vt = mmread("Vt.mtx")

Ac = U@S@Vt

print(f"{np.linalg.norm(A-Ac)}")

