# Setup
import csv
import numpy as np
import random

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

# Function for Mean Absolute Error
def meanAbsoluteError(weight, bias):
    sum = 0
    
    for i in range(len(y_np)):
        sum += abs(result(weight, x[i], bias) - y_np[i])
    
    return sum / len(y_np)

# One big function to save the amount of loops
def bigFunction(w, b):
    min = result(w, x[0], b) - y_np[0]
    max = result(w, x[0], b) - y_np[0]
    index_min = 0
    index_max = 0
    sum_1 = 0
    sum_2 = 0
    points_above = 0
    points_below = 0

    for i in range(len(y_np)):
        sum_1 += result(w, x[i], b) - y_np[i]
        sum_2 += abs(result(w, x[i], b) - y_np[i])

        if result(w, x[i], b) - y_np[i] > 0:
            points_below += 1
        elif result(w, x[i], b) - y_np[i] < 0:
            points_above += 1
        
        if abs(result(w, x[i], b) - y_np[i]) > max:
            max = result(w, x[i], b) - y_np[i]
            index_max = i
        if abs(result(w, x[i], b) - y_np[i]) < min:
            min = result(w, x[i], b) - y_np[i]
            index_min = i

    meanError = sum_1 / len(y_np)
    meanAbsoluteError = sum_2 / len(y_np)
    
    return min, max, index_min, index_max, points_above, points_below, meanError, meanAbsoluteError

# Function for printing algorithm output
def returnOutput(weight, bias):
    output = []
    
    for i in range(len(y_np)):
        output.append(result(weight, x[i], bias))
    
    return output
    
# Linear Regression algorithm
b = y[0]
w = (y[len(y)-1] - y[0]) / (x[len(x)-1] - x[0])

_, _, _, _, _, _, _, best = bigFunction(w, b)

while meanAbsoluteError(w, b) > 25:
    w_temp = w
    b_temp = b
    _, _, _, _, points_below_temp, points_above_temp, meanError_temp, meanAbsoluteError_temp = bigFunction(w,b)

    if meanError_temp < 0:
        b_temp += random.random() * w * 10
    elif meanError_temp > 0:
        b_temp -= random.random() * w * 10

    if points_below_temp > points_above_temp:
        w_temp -= random.random() * (w_temp / 10)
    elif points_below_temp < points_above_temp:
        w_temp += random.random() * (w_temp / 10)

    if meanAbsoluteError(w_temp, b_temp) < best:
        w = w_temp
        b = b_temp
        best = meanAbsoluteError(w, b)

# Output
def printOutput():
    print("")
    print("Weight: " + str(w))
    print("Bias: " + str(b))
    print("Equation: y = " + str(w) + "x+" + str(b))
    print("")
    print("Orginial: " + str(y_np.tolist()))
    print("")
    print("Predicted: " + str(returnOutput(w, b)))
    print("")
    
    min, max, index_min, index_max, _, _, meanError, meanAbsoluteError = bigFunction(w, b)
    print("Minimum difference is: " + str(min) + ", at node "+ str(index_min))
    print("Maximum difference is: " + str(max) + ", at node "+ str(index_max))
    
    print()
    print("")
    print("Mean Error: " + str(meanError))
    print("Mean Absolute Error: " + str(meanAbsoluteError))

printOutput()
