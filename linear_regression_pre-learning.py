# Setup
import csv
import numpy as np
import math

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

x = x_np.tolist()
y = y_np.tolist()

x.sort()
y.sort()

# Function for y=wx+b
def result(weight, x_value, bias):
    return weight*x_value + bias

# Function for Mean Error
def meanError(weight, bias):
    sum = 0
    
    for i in range(len(y_np)):
        sum += result(weight, bias) - y_np[i]
    
    return sum / len(y_np)
    

# Linear Regression algorithm
# clean_tries = 5

b = y[0]
w = (y[len(y)-1] - y[0]) / (x[len(x)-1] - x[0])

print(meanError(w, b))

# while clean_tries > 0:
# for i in x:
    