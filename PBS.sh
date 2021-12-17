#!/bin/bash

#PBS -l select=1:ncpus=1:mem=10G:ngpus=1
#PBS -N CNN_ECoG
#PBS -j oe

echo "working directory: " $PBS_O_WORKDIR
echo "omp thread num: " $OMP_NUM_THREADS
echo "ncpus: " $NCPUS
echo "cuda visible devices: " $CUDA_VISIBLE_DEVICES

start_time=`date +%s`

cd $PBS_O_WORKDIR
python test_smoothgrad.py

end_time=`date +%s`
run_time=$((end_time - start_time))

echo "start time: " $start_time
echo "end time: " $end_time
echo "run time: " $run_time
