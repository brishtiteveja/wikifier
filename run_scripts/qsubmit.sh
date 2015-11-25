#!/bin/sh
#PBS -l walltime=24:00:00
qsub -n -q kulkarni -p 1023 -l walltime=300:00:00,nodes=5:ppn=5:nodemem64gb $1 

