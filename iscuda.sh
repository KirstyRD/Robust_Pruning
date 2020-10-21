#!/bin/sh

#python3 iscuda.py
#modinfo nvidia | grep "^version:" | sed 's/^version: *//;'
#cat /etc/*release
#lspci -nnk | grep -i nvidia
#lspci -k | grep -EA3 'VGA|3D|Display'

#modinfo nvidia

#nvidia-smi



cd ~/.local/lib/python3.6/site-packages/torch/utils

python3 collect_env.py
