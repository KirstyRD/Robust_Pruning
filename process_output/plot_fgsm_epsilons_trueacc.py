import matplotlib.pyplot as plt
import sys

file_in  = sys.argv[1]
file_out = sys.argv[2]

import csv

x = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
y7 = []


with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        x.append(float(row[1]))
        y1.append(100*float(row[8])*float(row[6]))

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y2.append(100*float(row[9])*float(row[6]))

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y3.append(100*float(row[10])*float(row[6]))

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y4.append(100*float(row[11])*float(row[6]))

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y5.append(100*float(row[12])*float(row[6]))
        
with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y6.append(100*float(row[13])*float(row[6]))

with open(file_in,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    headers = next(plots)
    for row in plots:
        y7.append(100*float(row[14])*float(row[6]))
        
plt.plot(x,y1, label='Accuracy')
plt.plot(x,y2, label='e = 0.5')
plt.plot(x,y3, label='e = 1.0')
plt.plot(x,y4, label='e = 1.5')
plt.plot(x,y5, label='e = 2.0')
plt.plot(x,y4, label='e = 2.5')
plt.plot(x,y5, label='e = 3.0')
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title(sys.argv[3])
plt.legend()
plt.show()
plt.savefig(file_out)
