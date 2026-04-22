import numpy as np

# Input patterns (binary)
X = np.array([[1,0,1,0],
              [1,0,0,0],
              [0,1,0,1],
              [0,1,1,1]])

# Parameters
num_clusters = 2
weights = np.ones((num_clusters, X.shape[1]))

# Vigilance parameter
rho = 0.5

for i in range(len(X)):
    x = X[i]
    
    # Calculate match
    for j in range(num_clusters):
        match = np.sum(np.minimum(x, weights[j])) / np.sum(x)
        
        if match >= rho:
            # Update weights
            weights[j] = np.minimum(x, weights[j])
            print(f"Pattern {x} assigned to Cluster {j}")
            break