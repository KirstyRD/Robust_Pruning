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
        y1.append(100*float(row[4]))

with open(file_in2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y2.append(100*float(row[4]))

with open(file_in3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y3.append(100*float(row[4]))

with open(file_in4,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y4.append(100*float(row[4]))

with open(file_in5,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y5.append(100*float(row[4]))

        
plt.plot(x,y1, label='No DL2 term')
plt.plot(x,y2, label='DL2 weight 0.2')
plt.plot(x,y3, label='DL2 weight 0.1')
plt.plot(x,y4, label='DL2 weight 0.01')
plt.plot(x,y5, label='DL2 weight 0.05')
plt.xlabel('Number of Neurons')
plt.ylabel('Prediction Accuracy')
plt.title(sys.argv[7])
plt.legend()
plt.show()
plt.savefig(file_out)
