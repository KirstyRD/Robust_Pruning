#!/bin/bash

python3 main_6epsilons.py --dataset=CIFAR100 --model=resnet50 --pruning=True --tensorboard=True --log-interval=20 --pruning_config=./configs/CIFAR100_10_100.json --constraint="RobustnessDandR(eps1=7.8, eps2=2)" --print-after-epoch=0 --dl2-weight=0.1 --delay=0 --epochs=1000 --adv-after-epoch=20 --batch-size=32 --load_model="./CIFAR100_resnet50_256/best_CIFAR100_resnet50.weights" --name="CIFAR100_resnet50_128"
