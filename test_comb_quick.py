import subprocess as sp
from pathlib import Path
from glob import glob
import sys

comparisons = {}

for c_mtx in glob("*_c.mtx"):
    prefix = c_mtx.split("_")[0]
    comparisons[prefix] = {}
    comparisons[prefix]["c"] = Path(c_mtx)

for a_mtx in glob("*_a.mtx"):
    prefix = a_mtx.split("_")[0]
    comparisons[prefix]["a"] = Path(a_mtx)

for prefix, files in comparisons.items():
    if not "c" in files:
        print(f"No {prefix}_c.mtx file")
        continue
    assert files["c"].is_file()
    if not "a" in files:
        print(f"No {prefix}_a.mtx file")
        continue
    assert files["a"].is_file()
    cmd = ["python", "compare.py", str(files["c"]), str(files["a"])]
    p = sp.Popen(cmd, stdout=sp.PIPE)
    out, err = p.communicate()
    print(out.decode("utf-8"))
