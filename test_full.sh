#!/bin/bash

python svd_test.py
./full_svd A.mtx 10 5
./compare.py Aq1_11_c.mtx Aq1_11_a.mtx
./compare.py Aq1_12_c.mtx Aq1_12_a.mtx
./compare.py Vtq1_11_c.mtx Vtq1_11_a.mtx
./compare.py Vtq1_12_c.mtx Vtq1_12_a.mtx
