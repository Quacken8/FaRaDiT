#!/usr/bin/env bash

# runs radmc3d simulation on Tiger cluster
export PATH="/home/janoska/bin:$PATH"
rm output_old
mv output output_old

mpirun-test -n 16 -N1 -C24 --stdout=output radmc3d image incl 30 lambda 870 npix 500 countwrite 1000000 setthreads 12

#echo "Mctherm simulation on Tiger cluster done" | mail -s "Tiger simulation" janoska@sirrah.troja.mff.cuni.cz
