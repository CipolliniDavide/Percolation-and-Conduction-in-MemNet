#!/bin/bash

L=21
#crt=1
diag=0

# For good OC
#w_init='good_OC'
#svp='OutputGridAdiabatic_GoodOC'

# For Poor OC
w_init='None'
svp='OutputGridAdiabatic'

b_start=0
b_end=10


#for V in  10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
#for V in  70 80 90 #100 110 120 130 140 150
###
#do
#python ./create_simulation_data.py -Vb $V -diag $diag -svp $svp -b_start $b_start -b_end $b_end -lin_size $L -w_init $w_init
#done

crt_ds=0
#python ./create_plots_conductance.py -comp_ds $crt_ds -svp $svp -lin_size $L -w_init $w_init
python ./create_plots_entropy.py -comp_ds $crt_ds -svp $svp -lin_size $L -w_init $w_init
