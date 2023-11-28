#!/bin/bash

python svd_test.py
./full_svd A.mtx 10 5
python svd_test_check.py

./compare.py Sp_a.mtx Sp_c.mtx
./compare.py Up_a.mtx Up_c.mtx
./compare.py Vtp_a.mtx Vtp_c.mtx

#./compare.py Aq1_11_c.mtx Aq1_11_a.mtx
#./compare.py Aq1_12_c.mtx Aq1_12_a.mtx
#./compare.py Vtq1_11_c.mtx Vtq1_11_a.mtx
#./compare.py Vtq1_12_c.mtx Vtq1_12_a.mtx
