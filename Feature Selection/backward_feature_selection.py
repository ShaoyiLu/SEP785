from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np

data = np.load('data.npy')
labels = np.load('labels.npy')

classification_errors_backward = []

for n_features in range(10, 4, -1):
    lda = LinearDiscriminantAnalysis()

    if n_features < data.shape[1]:
        selector = SequentialFeatureSelector(lda, n_features_to_select=n_features, direction='backward')
        selector.fit(data, labels)
        reduced_data = selector.transform(data)
    else:
        reduced_data = data

    lda.fit(reduced_data, labels)

    accuracy = lda.score(reduced_data, labels)
    classification_error = 1 - accuracy
    classification_errors_backward.append(classification_error)

plt.figure(figsize=(10, 5))
plt.plot(range(10, 4, -1), classification_errors_backward, marker='o', color='blue', label="Classification Error")
plt.xlabel("Dimensions")
plt.ylabel("Classification Error")
plt.legend()
plt.title("Classification Error & Backward Selection")
plt.grid(True, which='both', alpha=0.5)
plt.show()
