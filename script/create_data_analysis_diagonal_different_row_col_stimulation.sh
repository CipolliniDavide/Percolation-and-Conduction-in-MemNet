#!/bin/bash

L=21
#crt=1
diag=1

# Location of source and ground electrodes
src=18
gnd=425


# For good OC
#w_init='good_OC'
#svp='DiagAdiab_diff_RowCol_goodOC'

# For Poor OC
w_init='None'
svp='DiagAdiab_diff_RowCol_poorOC'


b_start=1
b_end=50

for V in  10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
#for V in  150
###
do
python ./create_simulation_data_different_row_col_stimulation.py -src $src -gnd $gnd -Vb $V -diag $diag -svp $svp -b_start $b_start -b_end $b_end -lin_size $L -w_init $w_init
done

crt_ds=1
python ./create_plots_conductance.py -comp_ds $crt_ds -svp $svp -lin_size $L -w_init $w_init
python ./create_plots_entropy.py -comp_ds $crt_ds -svp $svp -lin_size $L -w_init $w_init
