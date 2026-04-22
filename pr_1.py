import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-10, 10, 100)

# 1. Step Function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# 2. Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. Tanh Function
def tanh(x):
    return np.tanh(x)

# 4. ReLU Function
def relu(x):
    return np.maximum(0, x)

# 5. Leaky ReLU Function
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

# Plotting all functions
plt.figure(figsize=(10, 8))

plt.plot(x, step_function(x), label="Step Function")
plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid()

plt.show()