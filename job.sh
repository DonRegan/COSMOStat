#!/bin/sh
#$ -N ZBOX2
#$ -M A.Eggemeier@sussex.ac.uk
#$ -m bea
#$ -cwd
#$ -pe openmpi_mixed_16 64
#$ -o info.dat


# queue:
#$ -q mps.q@@compute_amd_c6145_mps

# modules:
module add sge
module load intel
module load gcc/4.8.1
module load openmpi/gcc/64/1.7.3
module load fftw3/openmpi/gcc/64/3.3.4


export OMP_NUM_THREADS=1

mpirun -np $NSLOTS -npernode 16 ./linecorr
