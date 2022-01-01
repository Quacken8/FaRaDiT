#!/usr/bin/env bash

# runs radmc3d simulation on Tiger cluster
export PATH="/home/janoska/bin:$PATH"
rm output_old
mv output output_old

mpirun-test -n 16 --stdout=output radmc3d mctherm countwrite 1000000

echo "Mctherm simulation on Tiger cluster done" | mail -s "Tiger simulation" janoska@sirrah.troja.mff.cuni.cz
