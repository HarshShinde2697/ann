import numpy as np

# Input dataset (Example)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output dataset
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(1)
w1 = np.random.rand(2,3)   # Input to Hidden (2 inputs → 3 neurons)
w2 = np.random.rand(3,1)   # Hidden to Output

# Bias
b1 = np.random.rand(1,3)
b2 = np.random.rand(1,1)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def derivative(x):
    return x * (1 - x)

# Training
for i in range(10000):

    # Forward Propagation
    h = sigmoid(np.dot(X, w1) + b1)
    o = sigmoid(np.dot(h, w2) + b2)

    # Error
    error = y - o

    # Backpropagation
    d_o = error * derivative(o)
    d_h = d_o.dot(w2.T) * derivative(h)

    # Update weights
    w2 += h.T.dot(d_o) * 0.5
    w1 += X.T.dot(d_h) * 0.5

    # Update bias
    b2 += np.sum(d_o, axis=0)
    b1 += np.sum(d_h, axis=0)

# Output
print("Final Output:")
print(o)