#!/bin/sh
#PBS -l walltime=24:00:00
qsub -q kulkarni -p 1023 -l walltime=50:00:00,nodes=5:ppn=10:nodemem64gb $1 

