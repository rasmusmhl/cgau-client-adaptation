#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=8

for replicate in {1..20};
do
  for lsp in `seq 0.00 .20 1.00`; 
  do
    seed=$(($replicate+($1+1)*100))
    if $2
    then
      echo "Device $1, replicate $replicate, seed $seed, label_shuffle_percentage $lsp, with global_conditioning"
      python -u main.py --seed=$seed --label_shuffle_percentage=$lsp --experiment_group="cough_detector_v3" --global_conditioning=true>> label_shuffle_experiment.log_device$1
    else
      echo "Device $1, replicate $replicate, seed $seed, label_shuffle_percentage $lsp, no global_conditioning"
      python -u main.py --seed=$seed --label_shuffle_percentage=$lsp --experiment_group="cough_detector_v3" >> label_shuffle_experiment.log_device$1
    fi
  done
done
