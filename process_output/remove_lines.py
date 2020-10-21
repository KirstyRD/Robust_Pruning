import sys

lines = open(sys.argv[1], 'r').readlines()
search = 'epoch, neuron_units,'
for i, line in enumerate(lines):
    if search in line:
        break

if i < len(lines) - 1:
    with open(sys.argv[2], 'w') as f:
        f.write(''.join(lines[i + 1:]))
