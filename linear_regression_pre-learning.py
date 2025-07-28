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
        sum += result(weight, x[i], bias) - y_np[i]
    
    return sum / len(y_np)

# Function to find value with lowest and highest difference
def printMinandMax(weight, bias):
    min = result(weight, x[0], bias) - y[0]
    max = result(weight, x[0], bias) - y[0]
    index_min = 0
    index_max = 0
    
    for i in range(1, len(y_np)):
        if result(weight, x[i], bias) - y[i] > max:
            max = result(weight, x[i], bias) - y[i]
            index_max = i
        
        if result(weight, x[i], bias) - y[i] < min:
            min = result(weight, x[i], bias) - y[i]
            index_min = i
    
    print("Minimum difference is: " + str(min) + ", at node "+ str(index_min))
    print("Maximum difference is: " + str(max) + ", at node "+ str(index_max))

# Function for printing algorithm output
def returnOutput(weight, bias):
    output = []
    
    for i in range(len(y_np)):
        output.append(result(weight, x[i], bias))
    
    return output
    

# Linear Regression algorithm
b = y[0]
w = (y[len(y)-1] - y[0]) / (x[len(x)-1] - x[0])

# Output
print("")
print("Orginial: " + str(y))
print("")
print("Predicted: " + str(returnOutput(w, b)))
print("")
printMinandMax(w, b)
print("")
print("Mean Error: " + str(meanError(w, b)))
