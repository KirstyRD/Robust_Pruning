#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
dl2_weight=$5
dl2_constr=$6 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
batch_size=$7

config=$"./configs/${dataset}_${prune_per_iter}_${frequency}.json"

python3 main_6epsilons.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$frequency --pruning_config=$config --constraint="${dl2_constr}" --print-after-epoch=0 --dl2-weight=$dl2_weight --delay=0 --epochs=100 --adv-after-epoch=0 --batch-size=$batch_size --load_model="./CIFAR100/models/checkpoint_CIFAR100_resnet50_100_100_0.15_RobustnessDandR_eps1=7.8,_eps2=2.9__3052_4600_6.250000.weights" --name="CIFAR100_resnet50"

dl2_modified=${dl2_constr//[() ]/_}

cd process_output

python3 remove_header.py ../cuda/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight}_${dl2_modified}_FGSM.txt textfiles/print_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight}_${dl2_modified}_FGSM.txt

