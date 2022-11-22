#!/bin/bash

V=10
crt=1
diag=0
svp='Percolation'
b_start=0
b_end=20

#diag=1
#svp='Output'

python ./percolation_threshold.py -b 1000 --L 100 -svp $svp -crt_data 1
#python ./percolationThreshold_VoltageSweep.py -crt_data $crt -Vb $V -diag $diag -svp $svp -b_start $b_start -b_end $b_end

