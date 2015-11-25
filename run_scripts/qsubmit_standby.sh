#!/bin/sh
#PBS -l walltime=24:00:00
#qsub -q kulkarni -l walltime=300:00:00,nodes=5:ppn=16:nodemem64gb $1 
qsub -l walltime=3:00:00,nodes=5:ppn=16:nodemem64gb $1 

