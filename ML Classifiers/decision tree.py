import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Shaoyi Lu\Desktop\SEP 785\Course Project\NHANES_age_prediction.csv")

# Define features and target variable
features = ['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
target = 'age_group'

# Prepare data
X = data[features]
y = data[target]

# Convert target variable to numeric encoding if it is categorical
y = y.astype('category').cat.codes

# At least 75% data are used for train
train_size = min(1709, len(data))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

# Parameter search range
param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20, 50, 100],
    'min_samples_leaf': [1, 2, 4, 8, 16]
}

# Using KFold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Start time of train
start_time = time.time()

# Cross-validation and parameter tuning
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=kfold, scoring='accuracy')
clf.fit(X_train, y_train)

training_time = time.time() - start_time

# Best parameters
tuned_clf = clf.best_estimator_
print(f"Best parameters: {clf.best_params_}")

# Cross validation
cv_results = clf.cv_results_
cv_means = cv_results['mean_test_score']
cv_stds = cv_results['std_test_score']
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cv_means) + 1), cv_means, color='orange', yerr=cv_stds)
plt.xlabel("Parameters")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy (Decision Tree)")
plt.xticks(range(1, len(cv_means) + 1), rotation=45, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Cross Validation in Decision Tree.png")
plt.show()

# Time of Start
start_time = time.time()
y_pred = tuned_clf.predict(X_test)
testing_time = time.time() - start_time

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate ROC curve and AUC
y_pred_prob = tuned_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fpr, tpr, roc_auc = {}, {}, {}

# ROC Curves
plt.figure()
for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

    tuned_clf.fit(X_fold_train, y_fold_train)
    y_prob = tuned_clf.predict_proba(X_fold_test)[:, 1]

    fpr[i], tpr[i], _ = roc_curve(y_fold_test, y_prob)
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], label=f"Fold {i + 1} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Decision Tree)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ROC Curves in Decision Tree.png")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adult', 'Senior'])
cmd.plot(cmap="Reds")
plt.title("Confusion Matrix (Decision Tree)")
plt.savefig("Confusion Matrix in Decision Tree.png")
plt.tight_layout()
plt.show()

tree_rules = export_text(tuned_clf, feature_names=features)

print(f"Average Accuracy: {accuracy:.6f}")
print(f"Training time: {training_time:.6f} seconds")
print(f"Testing time: {testing_time:.6f} seconds")
print("\nConfusion Matrix:")
print(cm)
print("\nDecision Tree Rules (Formatted):")
for line in tree_rules.splitlines():
    formatted_line = line.replace("|", "├").replace("---", "→")
    print(formatted_line)

# Output:
# Best parameters: {'max_depth': 10, 'min_samples_leaf': 16, 'min_samples_split': 100}
# Average Accuracy: 0.827768
# Training time: 6.112074 seconds
# Testing time: 0.001337 seconds
#
# Confusion Matrix:
# [[465   9]
#  [ 89   6]]
#
# Decision Tree Rules (Formatted):
# ├→ LBXGLT <= 138.50
# ├   ├→ PAQ605 <= 1.50
# ├   ├   ├→ LBXGLU <= 100.50
# ├   ├   ├   ├→ BMXBMI <= 31.25
# ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├→ BMXBMI >  31.25
# ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├→ LBXGLU >  100.50
# ├   ├   ├   ├→ class: 0
# ├   ├→ PAQ605 >  1.50
# ├   ├   ├→ LBXGLT <= 90.50
# ├   ├   ├   ├→ LBXGLU <= 107.50
# ├   ├   ├   ├   ├→ LBXIN <= 5.89
# ├   ├   ├   ├   ├   ├→ BMXBMI <= 25.65
# ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├→ BMXBMI >  25.65
# ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├→ LBXIN >  5.89
# ├   ├   ├   ├   ├   ├→ BMXBMI <= 24.15
# ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├→ BMXBMI >  24.15
# ├   ├   ├   ├   ├   ├   ├→ RIAGENDR <= 1.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├   ├→ RIAGENDR >  1.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├→ LBXGLU >  107.50
# ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├→ LBXGLT >  90.50
# ├   ├   ├   ├→ LBXIN <= 10.71
# ├   ├   ├   ├   ├→ BMXBMI <= 25.55
# ├   ├   ├   ├   ├   ├→ LBXIN <= 4.98
# ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├→ LBXIN >  4.98
# ├   ├   ├   ├   ├   ├   ├→ LBXGLU <= 96.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├   ├→ LBXGLU >  96.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├→ BMXBMI >  25.55
# ├   ├   ├   ├   ├   ├→ LBXGLT <= 121.50
# ├   ├   ├   ├   ├   ├   ├→ BMXBMI <= 27.45
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├   ├→ BMXBMI >  27.45
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├→ LBXGLT >  121.50
# ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├→ LBXIN >  10.71
# ├   ├   ├   ├   ├→ LBXGLU <= 109.50
# ├   ├   ├   ├   ├   ├→ LBXGLT <= 109.50
# ├   ├   ├   ├   ├   ├   ├→ LBXGLT <= 95.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├   ├→ LBXGLT >  95.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├→ LBXGLT >  109.50
# ├   ├   ├   ├   ├   ├   ├→ LBXGLU <= 94.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├   ├   ├→ LBXGLU >  94.50
# ├   ├   ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├→ LBXGLU >  109.50
# ├   ├   ├   ├   ├   ├→ class: 0
# ├→ LBXGLT >  138.50
# ├   ├→ BMXBMI <= 41.20
# ├   ├   ├→ LBXGLU <= 99.50
# ├   ├   ├   ├→ class: 0
# ├   ├   ├→ LBXGLU >  99.50
# ├   ├   ├   ├→ LBXIN <= 6.45
# ├   ├   ├   ├   ├→ class: 1
# ├   ├   ├   ├→ LBXIN >  6.45
# ├   ├   ├   ├   ├→ LBXGLU <= 115.50
# ├   ├   ├   ├   ├   ├→ class: 0
# ├   ├   ├   ├   ├→ LBXGLU >  115.50
# ├   ├   ├   ├   ├   ├→ class: 1
# ├   ├→ BMXBMI >  41.20
# ├   ├   ├→ class: 0
