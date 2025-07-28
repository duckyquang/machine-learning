# Setup
import csv
import matplotlib.pyplot as plt
import numpy as np

# Build matrix and setup graph
matrix = []

with open("/Users/buno/Documents/coding/ai/machine-learning/models/car_specs_no_name.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    
    header = next(reader)
    for row in reader:
        float_row = [float(value) for value in row if value.strip() != ""]
        matrix.append(float_row)

matrix_np = np.array(matrix)

x_np = np.array(matrix_np[:,0])
y_np = np.array(matrix_np[:,1])

plt.scatter(x_np, y_np)
plt.show()