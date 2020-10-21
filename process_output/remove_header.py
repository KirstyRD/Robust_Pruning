import sys

lines = open(sys.argv[1], 'r').readlines()
search1 = 'epoch, neuron_units,'

copy = 0
for i, line in enumerate(lines):
    if search1 in line:
        break

if i < len(lines) - 1:
    with open(sys.argv[2], 'a') as f:
        f.write(''.join(lines[i:]))
