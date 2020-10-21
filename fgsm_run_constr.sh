#!/bin/bash

dataset=$1
network=$2
prune_per_iter=$3
frequency=$4
dl2_weight=$5
dl2_constr1=$6
dl2_constr2=$7 # RobustnessDandR(eps1=7.8, eps2=2.9)         
#attack=$7
batch_size=$8

config=$"./configs/${dataset}_${prune_per_iter}_${frequency}.json"

#cp CIFAR100/models/checkpoint_CIFAR100_resnet50_100_100_0.0_RobustnessDandR_eps1=7.8,_eps2=2.9__7552_26843545600000_100.000000.weights CIFAR100/cifar100_resnet50_best.weights

cp checkpoint_CIFAR100_resnet50_100_100_0.0_RobustnessDandR_eps1=7.8,_eps2=2.9__7552_107374182397200_100.000000.weights CIFAR100/cifar100_resnet50_best.weights

python3 main_6epsilons.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$frequency --pruning_config=$config --constraint="${dl2_constr1}" --print-after-epoch=0 --dl2-weight=$dl2_weight --delay=0 --epochs=1000 --adv-after-epoch=0 --batch-size=$batch_size --load_model="./CIFAR100_resnet50_16/batch128.weights" --name="CIFAR100_resnet50_16"#"./CIFAR100_resnet50_256/best_CIFAR100_resnet50.weights" --name="CIFAR100_resnet50_128"

#dl2_modified=${dl2_constr//[() ]/_}

#cd process_output

#python3 remove_header.py ../CIFAR100_resnet50_128/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight1}_${dl2_modified}_FGSM.txt textfiles/${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight1}_${dl2_modified}_FGSM.txt

#cd ..

#cp CIFAR100/models/checkpoint_CIFAR100_resnet50_100_100_0.0_RobustnessDandR_eps1=7.8,_eps2=2.9__7552_26843545600000_100.000000.weights CIFAR100/cifar100_resnet50_best.weights

cp checkpoint_CIFAR100_resnet50_100_100_0.0_RobustnessDandR_eps1=7.8,_eps2=2.9__7552_107374182397200_100.000000.weights CIFAR100/cifar100_resnet50_best.weights

python3 main_6epsilons.py --dataset=$dataset --model=$network --pruning=True --tensorboard=True --log-interval=$frequency --pruning_config=$config --constraint="${dl2_constr2}" --print-after-epoch=0 --dl2-weight=$dl2_weight --delay=0 --epochs=1000 --adv-after-epoch=0 --batch-size=$batch_size --load_model="./CIFAR100_resnet50_16/batch128.weights" --name="CIFAR100_resnet50_16"#"./CIFAR100_resnet50_256/best_CIFAR100_resnet50.weights" --name="CIFAR100_resnet50_128"

#dl2_modified=${dl2_constr//[() ]/_}

#cd process_output

#python3 remove_header.py ../CIFAR100_resnet50_128/log_${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight2}_${dl2_modified}_FGSM.txt textfiles/${dataset}_${network}_${prune_per_iter}_${frequency}_${dl2_weight2}_${dl2_modified}_FGSM.txt
