#!/usr/bin/env bash
# runs radmc3d simulation on Hilda cluster
export PATH="/home/janoska/bin:$PATH"
rm nohup_old.out
mv nohup.out nohup_old.out

nohup nice -16 radmc3d mctherm setthreads 8 countwrite 1000000

nohup nice -16 radmc3d image lambda 860 incl 60
