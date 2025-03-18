import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder

class NaiveBayesClassifier:
    def fit(self, X, y):  # X is feature, y is label
        self.classes = np.unique(y)
        self.mean = {}  # mean
        self.var = {}  # variance
        self.priors = {}  # prior probability

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9  # variance cannot be 0
            self.priors[c] = X_c.shape[0] / X.shape[0]

    # likelihood probability
    def _calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Gaussian distribution
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    # posterior probability
    def _calculate_posteriors(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])  # prior probability
            likelihood = np.sum(np.log(self._calculate_likelihood(c, x)))
            posteriors[c] = prior + likelihood  # posterior probability
        return posteriors

    # predict sample
    def predict(self, X):
        predictions = []
        for sample in X:
            posteriors = self._calculate_posteriors(sample)
            predictions.append(max(posteriors, key=posteriors.get))  # select the class with the largest posterior prob
        return np.array(predictions)

    # predict probability
    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            posteriors = self._calculate_posteriors(sample)

            probs = {}
            for c in posteriors:
                probs[c] = np.exp(posteriors[c])

            total_prob = 0
            for c in probs:
                total_prob += probs[c]  # prob for every class

            normalized_probs = {}
            for c in probs:
                normalized_probs[c] = probs[c] / total_prob

            probabilities.append(normalized_probs)
        return probabilities

data = pd.read_csv(r"C:\Users\Shaoyi Lu\Desktop\SEP 785\Course Project\NHANES_age_prediction.csv")

if data.isnull().sum().any():
    data.fillna(data.mean(), inplace=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['age_group'])
X = data[['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']].values

# Use KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# Cross-validation and parameter tuning
fold_idx = 1
for train_idx, test_idx in kfold.split(X):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train_fold, y_train_fold)
    y_pred_fold = nb_classifier.predict(X_test_fold)
    accuracy = np.mean(y_pred_fold == y_test_fold)
    cv_scores.append(accuracy)
    print(f"Fold {fold_idx}: Accuracy = {accuracy:.4f}")
    fold_idx += 1

# Cross validation results
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='green')
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross Validation (Naïve Bayes)")
plt.xticks(range(1, len(cv_scores) + 1))
plt.ylim(0, 1)
for i, score in enumerate(cv_scores):
    plt.text(i + 1, score + 0.02, f"{score:.2f}", ha='center', fontsize=10, color='black')
plt.tight_layout()
plt.savefig("Cross Validation in Naïve Bayes.png")
plt.show()

print(f"Average Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1709, random_state=42)

# initialize
nb_classifier = NaiveBayesClassifier()

# time of train
start_time = time.time()
nb_classifier.fit(X_train, y_train)
training_time = time.time() - start_time

# time of test
start_time = time.time()
y_pred = nb_classifier.predict(X_test)
testing_time = time.time() - start_time

# evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

start_time = time.perf_counter()
nb_classifier.fit(X_train, y_train)
train_time = time.perf_counter() - start_time
print(f"Training Time: {train_time:.6f} seconds")
print(f"Testing Time: {testing_time:.4f} seconds")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Naïve Bayes)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="red")
plt.savefig("Confusion Matrix in Naïve Bayes.png")
plt.tight_layout()
plt.show()

# ROC curves
kf = KFold(n_splits=5)
fpr, tpr, roc_auc = {}, {}, {}

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    nb_classifier.fit(X_train_fold, y_train_fold)

    y_prob = nb_classifier.predict_proba(X_test_fold)
    prob_1 = []
    for prob in y_prob:
        prob_1.append(prob[1])
    y_prob = prob_1

    fpr[i], tpr[i], _ = roc_curve(y_test_fold, y_prob, pos_label=1)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(roc_auc)):
    plt.plot(fpr[i], tpr[i], label=f'Fold {i + 1} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Naïve Bayes)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("ROC Curves in Naïve Bayes.png")
plt.show()

# Output:
# Fold 1: Accuracy = 0.8333
# Fold 2: Accuracy = 0.8355
# Fold 3: Accuracy = 0.8684
# Fold 4: Accuracy = 0.8352
# Fold 5: Accuracy = 0.8022
# Average Accuracy: 0.8349
# Standard Deviation of CV Accuracy: 0.0210
# Training Time: 0.000523 seconds
# Testing Time: 0.0165 seconds
# Confusion Matrix:
# [[456  18]
#  [ 80  15]]
#
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.85      0.96      0.90       474
#            1       0.45      0.16      0.23        95
#
#     accuracy                           0.83       569
#    macro avg       0.65      0.56      0.57       569
# weighted avg       0.78      0.83      0.79       569
