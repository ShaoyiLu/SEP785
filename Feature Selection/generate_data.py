import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

mean_A = np.array([2.5, 3.6, 1.2, 7.4, 0.9, 5.5, 3.1, 6.8, 2.7, 4.0])
mean_B = np.array([2.0, 3.1, 1.1, 7.0, 0.8, 5.3, 3.0, 6.6, 2.6, 3.9])

cov_matrix = np.array([
    [1, 0.5, 0.3, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
    [0.3, 0.5, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0.5, 0.3, 0, 0, 0, 0],
    [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
    [0, 0, 0, 0.3, 0.5, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0.5, 0.3, 0],
    [0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0],
    [0, 0, 0, 0, 0, 0, 0.3, 0.5, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

class_A = np.random.multivariate_normal(mean_A, cov_matrix, 100)
class_B = np.random.multivariate_normal(mean_B, cov_matrix, 100)

data = np.vstack((class_A, class_B))
labels = np.array([0] * 100 + [1] * 100)

plt.figure(figsize=(8, 6))
plt.scatter(class_A[:, 0], class_A[:, 1], label='Class A', alpha=0.6, color='green')
plt.scatter(class_B[:, 0], class_B[:, 1], label='Class B', alpha=0.6, color='orange')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.title("Scatter Plot of First Two Dimensions")
plt.show()

np.save('data.npy', data)
np.save('labels.npy', labels)
