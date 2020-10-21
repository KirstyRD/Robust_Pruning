#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
dl2_weight1=$5
dl2_weight2=$6
dl2_constr=$7 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
log_frequency=1
batch_size=$8

config=$"./configs/${dataset}_${prune_per_iter}_${frequency}.json"


python3 main_6epsilons.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$log_frequency --pruning_config=$config --constraint="${dl2_constr}" --print-after-epoch=0 --dl2-weight=$dl2_weight1 --delay=10 --epochs=1000 --adv-after-epoch=10 --batch-size=$batch_size --load_model="" --name="CIFAR10_lenet" #"./CIFAR100_resnet50_16/batch128.weights" --name="CIFAR100_resnet50_16"


python3 main_6epsilons.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$frequency --pruning_config=$config --constraint="${dl2_constr}" --print-after-epoch=0 --dl2-weight=$dl2_weight2 --delay=0 --epochs=1000 --adv-after-epoch=0 --batch-size=$batch_size --load_model="" --name="CIFAR10_lenet"  

dl2_modified=${dl2_constr//[() ]/_}

cd process_output

python3 remove_header.py ../CIFAR10_lenet/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight1}_${dl2_modified}_FGSM.txt text
files/${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight1}_${dl2_modified}_64.txt


dl2_modified=${dl2_constr//[() ]/_}

python3 remove_header.py ../CIFAR10_lenet/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight2}_${dl2_modified}_FGSM.txt textfiles/${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight2}_${dl2_modified}_64.txt
