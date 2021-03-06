#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
dl2_weight=$5
dl2_constr=$6 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
restarts=$7
log_file=$"process_output/text/${dataset}_${network}_$3_$4_$5_RobustnessG_eps=0.3,_delta=0.52__64_r${restarts}.txt"
log_frequency=1
batch_size=$8
starting_line=0


config=$"./configs/${dataset}_${prune_per_iter}_${frequency}.json"


python3 test_models.py --dataset=$dataset --model=$network --tensorboard=True  --pruning_config=$config --constraint="${dl2_constr}" --dl2-weight=$dl2_weight --batch-size=$batch_size --name="CIFAR100_res_64_advset" --checkpoint_folder="CIFAR100_res_64" --log_file=$log_file  --starting_line=$starting_line --restarts=$restarts

