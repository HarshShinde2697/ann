import numpy as np
import matplotlib.pyplot as plt

# Input data (2D points)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output labels (AND function)
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
lr = 0.1

# Training
for epoch in range(10):
    for i in range(len(X)):
        x = X[i]
        target = y[i]

        # Prediction
        net = np.dot(x, w) + b
        output = 1 if net >= 0 else 0

        # Update rule
        error = target - output
        w = w + lr * error * x
        b = b + lr * error

# Plotting
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')

# Decision boundary
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = -(w[0]*x_vals + b) / w[1]

plt.plot(x_vals, y_vals)

plt.title("Perceptron Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()

plt.show()