#!/bin/bash

python3 main_run.py --name=runs/resnet50/resnet50_prune72 --dataset=Imagenet --lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=25 --batch-size=256 --pruning=True --seed=0 --model=resnet50 --load_model=./models/pretrained/resnet50-19c8e357.pth --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=True --pruning-method=22  --data=/imagenet/ --no_grad_clip=True --pruning_config=./configs/imagenet_resnet50_prune72.json --constraint="RobustnessDandR(eps1=7.8, eps2=2.9)" --print-after-epoch=0 --dl2-weight=0.01 --delay=0 --epochs=20 --adv-after-epoch=0
