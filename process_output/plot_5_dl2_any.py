import matplotlib.pyplot as plt
import sys

file_in1 = sys.argv[1]
file_in2 = sys.argv[2]
file_in3 = sys.argv[3]
file_in4 = sys.argv[4]
file_in5 = sys.argv[5]
file_in6 = sys.argv[6]
file_out = sys.argv[7]
column   = sys.argv[10]

import csv

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []


with open(file_in1,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x1.append(float(row[1]))
        y1.append(float(row[int(column)]))

with open(file_in2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x2.append(float(row[1]))
        y2.append(float(row[int(column)]))

with open(file_in3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x3.append(float(row[1]))
        y3.append(float(row[int(column)]))

with open(file_in4,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x4.append(float(row[1]))
        y4.append(float(row[int(column)]))

with open(file_in5,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x5.append(float(row[1]))
        y5.append(float(row[int(column)]))

with open(file_in6,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x6.append(float(row[1]))
        y6.append(float(row[int(column)]))

plt.plot(x1,y1, label='dl2 weight 0.0', linewidth=0.5)
plt.plot(x2,y2, label='dl2 weight 0.005', linewidth=0.5)
plt.plot(x3,y3, label='dl2 weight 0.01', linewidth=0.5)
plt.plot(x4,y4, label='dl2 weight 0.02', linewidth=0.5)
plt.plot(x5,y5, label='dl2 weight 0.05', linewidth=0.5)
plt.plot(x6,y6, label='dl2 weight 0.1', linewidth=0.5)
plt.xlabel('Number of Parameters')
plt.ylabel(sys.argv[9])
plt.title(sys.argv[8])
plt.legend()
plt.show()
plt.savefig(file_out)
