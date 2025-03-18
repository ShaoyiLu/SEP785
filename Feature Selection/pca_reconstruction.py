from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = np.load('data.npy')
labels = np.load('labels.npy')

pca = PCA(n_components=10)
data_pca = pca.fit_transform(data)

reconstruction_errors = []
classification_errors = []

for components in range(10, 4, -1):
    pca_reduced = PCA(n_components=components)
    reduced_data = pca_reduced.fit_transform(data)

    reconstructed_data = pca_reduced.inverse_transform(reduced_data)
    mse = mean_squared_error(data, reconstructed_data)
    reconstruction_errors.append(mse)

    lda = LinearDiscriminantAnalysis()
    lda.fit(reduced_data, labels)
    predictions = lda.predict(reduced_data)
    classification_error = 1 - accuracy_score(labels, predictions)
    classification_errors.append(classification_error)

plt.figure(figsize=(10, 5))
plt.plot(range(10, 4, -1), reconstruction_errors, marker='o', color='gold', label="Reconstruction MSE")
plt.xlabel("PCA Components")
plt.ylabel("MSE")
plt.legend()
plt.title("Reconstruction Error & PCA Components")
plt.grid(True, which='both', alpha=0.5)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(10, 4, -1), classification_errors, marker='o', color='purple', label="Classification Error")
plt.xlabel("PCA Components")
plt.ylabel("Classification Error")
plt.legend()
plt.title("Classification Error & PCA Components")
plt.grid(True, which='both', alpha=0.5)
plt.show()
