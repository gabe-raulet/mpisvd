#!/usr/bin/env python

import sys
import numpy as np
from scipy.io import mmread

def main(Afile, Bfile):
    A = mmread(Afile)
    B = mmread(Bfile)

    if A.shape != B.shape:
        sys.stdout.write("Matrices '{}' and '{}' have different shapes\n".format(Afile, Bfile))
        sys.stdout.flush()
        return 1

    if np.allclose(A, B):
        sys.stdout.write("Matrices '{}' and '{}' are equal\n".format(Afile, Bfile))
        sys.stdout.flush()
    else:
        sys.stdout.write("Matrices '{}' and '{}' are not equal\n".format(Afile, Bfile))
        sys.stdout.flush()

    sys.stdout.write(f"{np.linalg.norm(A-B):.10e}\n")
    sys.stdout.flush()

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python {} <A.mtx> <B.mtx>\n".format(sys.argv[0]))
        sys.stderr.flush()
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
