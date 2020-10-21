import sys
# argv2 = normal
# argv3 = dl2 print

lines = open(sys.argv[1], 'r').readlines()
search1 = 'epoch, neuron_units,'


for i, line in enumerate(lines):
    if i == 0:
        with open(sys.argv[2], 'a') as f:
            f.write(line)
    if i % 2 == 1:
        with open(sys.argv[2], 'a') as f:
            f.write(line)
    if i % 2 == 0:
        with open(sys.argv[3], 'a') as f:
            f.write(line)
