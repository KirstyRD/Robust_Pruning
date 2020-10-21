import matplotlib.pyplot as plt
import sys

file_in = sys.argv[1]
file_out = sys.argv[2]

import csv

x = []
y1 = []
y2 = []

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    iter = 0
    for row in plots:
        iter = iter + 1
        x.append(iter)
        y1.append(float(row[4]))
        y2.append(100*float(row[7]))

plt.plot(x,y1, label='Prediction Accuracy')
plt.plot(x,y2, label='Constraint Accuracy')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.title(sys.argv[3])
plt.legend()
plt.show()
plt.savefig(file_out)
