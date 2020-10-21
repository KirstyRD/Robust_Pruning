import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, sys

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

import numpy as np
import itertools


#loader batchsize=1
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor()#, transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True)

num_steps = 0

#create lists for 10 classes
inputs = []
classes = [0,1,2,3,4,5,6,7,8,9] #mnist

for i in range(0,len(classes)):
    inputs_n = []
    inputs.append(inputs_n)

#forall inputs in loader
for data, target in test_loader:
    num_steps = num_steps + 1

    # separate into classes
    i = int(target)
    inputs[i].append(data.flatten())

    #if num_steps >= 30:
        #break

centroids = []    
distances = []

for i in range(0,len(classes)):
    dist_n = []
    distances.append(dist_n)
    

for label in range(len(classes)):
    for a, b in itertools.combinations(inputs[label], 2):
        # add to the list
        distances[label].append(float(torch.norm((a - b)#.view(1,-1), dim=1
        )) )

    np_inputs = []
    for inp in range(len(inputs[label])):
        np_inputs.append(inputs[label][inp].numpy())
    centroids.append( np.mean(np_inputs, axis=0))


for label in range(len(classes)):

    #calculate min,max,avg,stddev
    print(np.mean(distances[label]))

print("centroids")

for a, b in itertools.combinations(classes, 2):
    print(a, ", ", b)
    print( float(torch.norm((torch.from_numpy(centroids[a]) - torch.from_numpy(centroids[b]) )#.view(1,-1) , dim=1
    )) )
