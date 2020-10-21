import matplotlib.pyplot as plt
import sys
import csv

num_files = sys.argv[1]

file_in = []

for i in range(num_files):
    file_in[i] = sys.argv[i+2]

plot_row = sys.argv[num_files+2]
y_axis   = sys.argv[num_files+3]
title_   = sys.argv[num_files+4]
file_out = sys.argv[num_files+5]

legend = []

for i in range(num_files):
    legend[i] = sys.argv[i+num_files+6]

x = []
y = []

for i in range(num_files):
    x[i] = []
    y[i] = []

for i in range(num_files):
    with open(file_in[i],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        headers = next(plots)
        for row in plots:
            x[i].append(float(row[1]))
            y[i].append(100*float(row[plot_row]))

for i in range(num_files):
    plt.plot(x[i],y[i], label=legend[i], linewidth=0.5)

plt.xlabel('Number of Parameters')
plt.ylabel(y_axis)
plt.title(title_)
plt.legend()
plt.show()
plt.savefig(file_out)
