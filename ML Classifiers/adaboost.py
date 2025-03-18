import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Shaoyi Lu\Desktop\SEP 785\Course Project\NHANES_age_prediction.csv")

features = ['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
target = 'age_group'

X = data[features]
y = data[target]

# Convert target variable to numeric encoding
y = y.astype('category').cat.codes

# At least 75% datas are used for train
train_size = min(1709, len(data))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

# Custom AdaBoost implementation
def adaboost_verbose(X_train, y_train, X_test, y_test, n_estimators=50):
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alphas = []

    for estimator in range(n_estimators):
        # Initialize weak learner
        weak_learner = DecisionTreeClassifier(max_depth=2, min_samples_split=5, min_samples_leaf=2, random_state=42)
        weak_learner.fit(X_train, y_train, sample_weight=weights)

        # Prediction results and errors
        y_pred_train = weak_learner.predict(X_train)
        error = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)

        # Stop training if error rate = 0
        if error == 0:
            alpha = 1
            classifiers.append(weak_learner)
            alphas.append(alpha)
            break

        # Weight for weak learner
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))

        # Update sample weights
        weights = weights * np.exp(-alpha * (2 * y_train - 1) * (2 * y_pred_train - 1))
        weights /= np.sum(weights)

        classifiers.append(weak_learner)
        alphas.append(alpha)

    # Testing
    pred = np.zeros(X_test.shape[0])
    for alpha, clf in zip(alphas, classifiers):
        pred += alpha * (2 * clf.predict(X_test) - 1)

    # Prediction
    y_pred_final = (pred > 0).astype(int)
    return y_pred_final, classifiers, alphas, pred

# Start time of train
start_time = time.time()
y_pred, classifiers, alphas, pred_scores = adaboost_verbose(X_train, y_train, X_test, y_test, n_estimators=50)
training_time = time.time() - start_time

# Test performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

start_testing_time = time.time()
y_pred, classifiers, alphas, pred_scores = adaboost_verbose(X_train, y_train, X_test, y_test, n_estimators=50)
testing_time = time.time() - start_testing_time

print(f"Accuracy: {accuracy:.4f}")
print(f"Training Time: {training_time:.6f} seconds")
print(f"Testing Time: {testing_time:.6f} seconds")
print(f"Confusion Matrix:\n{cm}")

# Confusion Matrix
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adult', 'Senior'])
cmd.plot(cmap="Purples")
plt.title("Confusion Matrix (AdaBoost)")
plt.savefig("Confusion Matrix in AdaBoost.png")
plt.tight_layout()
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_accuracies = []
plt.figure()

for i, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Retrain AdaBoost
    _, _, _, val_pred_scores = adaboost_verbose(X_fold_train, y_fold_train, X_fold_val, y_fold_val, n_estimators=50)
    val_pred = (val_pred_scores > 0).astype(int)
    cross_val_accuracies.append(accuracy_score(y_fold_val, val_pred))

    # ROC curve for each fold
    fpr, tpr, _ = roc_curve(y_fold_val, val_pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")

# ROC curves
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (AdaBoost)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ROC Curves in AdaBoost.png")
plt.show()

# Cross validation
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cross_val_accuracies) + 1), cross_val_accuracies, color='yellow')
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross Validation (AdaBoost)")
plt.ylim(0, 1)
for i, acc in enumerate(cross_val_accuracies):
    plt.text(i + 1, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("Cross Validation in AdaBoost.png")
plt.show()

print("\nCross Validation Results:")
print(f"Average Accuracy: {np.mean(cross_val_accuracies):.4f}")
print(f"Standard Deviation: {np.std(cross_val_accuracies):.4f}")


# Output:
# Accuracy: 0.8489
# Training Time: 0.270286 seconds
# Testing Time: 0.269494 seconds
# Confusion Matrix:
# [[462  12]
#  [ 74  21]]
#
# Cross Validation Results:
# Average Accuracy: 0.8408
# Standard Deviation: 0.0226
