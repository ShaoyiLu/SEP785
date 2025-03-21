import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Shaoyi Lu/Desktop/SEP 785/Assignment 1/multNormal.xlsx"
data = pd.read_excel(path, header=None)

covariance = []
variance = []

for i in range(1, 11):
    example = data.iloc[:, [(i-1)*2, (i-1)*2 + 1]]

    cov_matrix = np.cov(example, rowvar=False)
    print(f"Covariance Matrix for Example {i}:")
    print(cov_matrix)
    print()

    covariance.append(cov_matrix[0, 1])
    variance.append([cov_matrix[0, 0], cov_matrix[1, 1]])

    plt.scatter(example.iloc[:, 0], example.iloc[:, 1], alpha=0.5)
    plt.title(f"2D Scatter Plot for Example {i}")
    plt.xlabel(f"Column {(i-1)*2 + 1}")
    plt.ylabel(f"Column {(i-1)*2 + 2}")
    plt.grid(True)
    plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), covariance, marker='x', label='Covariance Value')
plt.title('Covariance value for each example')
plt.xlabel('Example')
plt.ylabel('Covariance Value')
plt.grid(True)
plt.legend()
plt.show()

variances = np.array(variance)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), variances[:, 0], marker='^', color='red', label='Variance of Column 1')
plt.plot(range(1, 11), variances[:, 1], marker='v', color='green', label='Variance of Column 2')
plt.title('Variance for the first column and second column for each example')
plt.xlabel('Example')
plt.ylabel('Variance')
plt.grid(True)
plt.legend()
plt.show()

# Covariance Matrix for Example 1:
# [[10.34020531 -0.01203439]
#  [-0.01203439  2.06085007]]
#
# Covariance Matrix for Example 2:
# [[9.73385592 3.90880004]
#  [3.90880004 2.48255605]]
#
# Covariance Matrix for Example 3:
# [[9.81331096 5.9219742 ]
#  [5.9219742  4.24147055]]
#
# Covariance Matrix for Example 4:
# [[8.71864832 6.33745747]
#  [6.33745747 5.11128219]]
#
# Covariance Matrix for Example 5:
# [[8.08741481 7.13202774]
#  [7.13202774 6.80509819]]
#
# Covariance Matrix for Example 6:
# [[7.12792232 7.45150027]
#  [7.45150027 8.41154389]]
#
# Covariance Matrix for Example 7:
# [[5.11175555 6.26160543]
#  [6.26160543 8.60036264]]
#
# Covariance Matrix for Example 8:
# [[3.81863159 5.42111292]
#  [5.42111292 9.12946713]]
#
# Covariance Matrix for Example 9:
# [[2.6227809  4.00206938]
#  [4.00206938 9.34683606]]
#
# Covariance Matrix for Example 10:
# [[ 1.9864565  -0.09367179]
#  [-0.09367179  9.54814947]]

# From the 2D scatter plots, the geometric changes in each example are reflected in the gradual
# variation in the distribution of data points. From Graph 1 to Graph 6, the linear arrangement
# of data points gradually strengthens, indicating that the covariance is increasing. As the
# examples progress, the data points begin to concentrate along a line close to the diagonal,
# signifying a significant enhancement in the correlation between the two variables, which is
# also a geometric representation of the increased covariance. However, from Graph 6 to Graph 10,
# the linear arrangement of data points starts to weaken, indicating that the covariance is
# decreasing and the correlation between the variables is gradually diminishing. Geometrically,
# the data points transition from a scattered random distribution to a more linear arrangement,
# and then begin to disperse again. This suggests that the geometric shape of the data points is
# influenced by some linear transformation.
# Linear transformation is the key reason behind this geometric change. The covariance matrices
# are generated by applying rotation and scaling transformations to the data points. Rotation can
# alter the directional relationships between the variables, while scaling transformations can
# change the strength of the correlation between the variables.
