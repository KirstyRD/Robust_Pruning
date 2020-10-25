#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
load_model=$5
dl2_constr=$6 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
batch_size=$7

config=$"./configs/${dataset}_train.json"

python3 main_gc.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$frequency --pruning_config=$config --constraint="${dl2_constr}" --print-after-epoch=0 --dl2-weight=0 --delay=0 --epochs=200 --adv-after-epoch=100000000000 --batch-size=$batch_size --load_model=$load_model --name="CIFAR100_res" --lr=0.2 --lr-decay-every=60 --lr-decay-scalar=0.2

dl2_modified=${dl2_constr//[() ]/_}

cd process_output

python3 remove_header.py ../CIFAR100_resnet50_256/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight}_${dl2_modified}_FGSM.txt textfiles/print_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight}_${dl2_modified}_train.txt

