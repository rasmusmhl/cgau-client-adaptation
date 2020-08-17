#!/bin/sh
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 # replicates
do
   for j in 0.0 0.2 0.4 0.6 0.8 1.0
   do
   echo "Replicate $i with label_shuffle_percentage $j"
   python -u train_federated_model.py --seed=$i --label_shuffle_percentage=$j --global_conditioning=true --experiment_group="CIFAR10" > label_shuffle_experiment.log1 2>&1
   
   python -u train_federated_model.py --seed=$i --label_shuffle_percentage=$j --experiment_group="CIFAR10" > label_shuffle_experiment.log1 2>&1
   echo "global_conditioning"
   
   done
   
   python -u train_federated_model.py --seed=$i --all_client_data_centrally=true --experiment_group="CIFAR10"> label_shuffle_experiment.log1 2>&1
done
