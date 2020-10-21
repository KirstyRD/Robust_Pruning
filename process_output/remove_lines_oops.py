import sys

lines = open(sys.argv[1], 'r').readlines()
search = 'tensor'
for i, line in enumerate(lines):
    if search in line:
        pass
    else:
        with open(sys.argv[2], 'a') as f:
            f.write(line)
