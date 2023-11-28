#!/bin/bash

# python seed_test.py m n r p q cond damp
#
python seed_test.py 256 128 128 10 3 100 2
./seed_test A.mtx 10 3
./compare.py Aseed.mtx Aseed_test.mtx
./compare.py Vtseed.mtx Vtseed_test.mtx
#./compare.py Acombs.mtx Acombs_test.mtx
#./compare.py Vtcombs.mtx Vtcombs_test.mtx

#python seed_test.py 1024 256 256 10 3 100 2
#./seed_test A.mtx 10 3
#./compare.py Aseed.mtx Aseed_test.mtx
#./compare.py Vtseed.mtx Vtseed_test.mtx

#python seed_test.py 1024 512 512 15 5 100 2
#./seed_test A.mtx 15 5
#./compare.py Aseed.mtx Aseed_test.mtx
#./compare.py Vtseed.mtx Vtseed_test.mtx

#python seed_test.py 4096 1024 1024 5 5 100 2
#./seed_test A.mtx 5 5
#./compare.py Aseed.mtx Aseed_test.mtx
#./compare.py Vtseed.mtx Vtseed_test.mtx
