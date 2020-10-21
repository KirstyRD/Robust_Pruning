import matplotlib.pyplot as plt
import sys

file_in1 = sys.argv[1]
file_in2 = sys.argv[2]
file_in3 = sys.argv[3]
file_in4 = sys.argv[4]
file_in5 = sys.argv[5]
file_out = sys.argv[6]

import csv

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []


with open(file_in1,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x1.append(float(row[1]))
        y1.append(100*float(row[13]))

with open(file_in2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x2.append(float(row[1]))
        y2.append(100*float(row[13]))
        
with open(file_in3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x3.append(float(row[1]))
        y3.append(100*float(row[13]))

with open(file_in4,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x4.append(float(row[1]))
        y4.append(100*float(row[13]))

with open(file_in5,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x5.append(float(row[1]))
        y5.append(100*float(row[13]))

        
plt.plot(x1,y1, label='10 iterations', linewidth=0.5)
plt.plot(x2,y2, label='20 iterations', linewidth=0.5)
plt.plot(x3,y3, label='30 iterations', linewidth=0.5)
plt.plot(x4,y4, label='40 iterations', linewidth=0.5)
plt.plot(x5,y5, label='50 iterations', linewidth=0.5)
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title(sys.argv[7])
plt.legend()
plt.show()
plt.savefig(file_out)
