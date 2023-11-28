#!/bin/bash

python comb_test.py
./comb_test
./compare.py Ak1_lj_correct.mtx Ak1_lj_attempt.mtx
./compare.py Vtk1_lj_correct.mtx Vtk1_lj_attempt.mtx

