import pandas as pd

path = "C:/Users/Shaoyi Lu/Desktop/SEP 785/Assignment 1/dataA.xlsx"

data = pd.read_excel(path, header=None)

mean = data.mean()
cov = data.cov()

print("Mean of each attribute:")
print(mean)
print("\nCovariance matrix:")
print(cov)

# Answer:

# Mean of each attribute:
# 0    0.347502
# 1    1.025637
# 2    0.801221

# Covariance matrix:
#           0         1         2
# 0  4.070489  0.150202  0.262084
# 1  0.150202  2.563071  0.014686
# 2  0.262084  0.014686  3.183212
