import matplotlib.pyplot as plt
import sys

file_in1 = sys.argv[1]
file_in2 = sys.argv[2]
file_in3 = sys.argv[3]
file_in4 = sys.argv[4]
file_in5 = sys.argv[5]
file_out = sys.argv[6]

import csv

x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []


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

with open(file_in5,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y5.append(100*float(row[7]))
        
plt.plot(x,y1, label=sys.argv[7])
plt.plot(x,y2, label=sys.argv[8])
plt.plot(x,y3, label=sys.argv[9])
plt.plot(x,y4, label=sys.argv[10])
plt.plot(x,y5, label=sys.argv[11])
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title(sys.argv[12])
plt.legend()
plt.show()
plt.savefig(file_out)
