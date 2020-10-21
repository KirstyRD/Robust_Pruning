import matplotlib.pyplot as plt
import sys

file_in1 = sys.argv[1]
file_in2 = sys.argv[2]
file_in3 = sys.argv[3]
file_in4 = sys.argv[4]
file_out = sys.argv[5]

import csv

x = []
y1 = []
y2 = []
y3 = []
y4 = []


with open(file_in1,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x.append(float(row[1]))
        y1.append(100*float(row[7]))

with open(file_in2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y2.append(100*float(row[7]))

with open(file_in3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y3.append(100*float(row[7]))

with open(file_in4,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y4.append(100*float(row[7]))

        
plt.plot(x,y1, label='e2 = 2')
plt.plot(x,y2, label='e2 = 4')
plt.plot(x,y3, label='e3 = 6')
plt.plot(x,y4, label='e4 = 8')
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title(sys.argv[6])
plt.legend()
plt.show()
plt.savefig(file_out)
