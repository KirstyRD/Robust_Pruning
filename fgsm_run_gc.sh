#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
dl2_weight1=$5
#dl2_weight2=$6
dl2_constr=$6 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
log_frequency=$4
batch_size=$7

config=$"./configs/${dataset}_${prune_per_iter}_${frequency}.json"
folder=$"CIFAR100_res_64"

python3 main_gc.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$log_frequency --pruning_config=$config --constraint="${dl2_constr}" --print-after-epoch=0 --dl2-weight=$dl2_weight1 --delay=0 --epochs=1000 --adv-after-epoch=0 --batch-size=$batch_size --load_model="./CIFAR100_res/best_model.weights" --name=$folder

cd process_output

python3 remove_header.py ../${folder}/log_${dataset}_${network}_$3_$4_$5_RobustnessG_eps=0.3,_delta=0.52__$8.txt log_${dataset}_${network}_$3_$4_$5_RobustnessG_eps=0.3,_delta=0.52__$8.txt
