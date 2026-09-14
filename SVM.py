# SVM Classification on Iris Dataset
# 80% Training, 20% Testing
# Linear Kernel

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Iris Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------------------------------
# 2. Split Dataset
#    80% Training and 20% Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 4. Create SVM Model
#    Linear Kernel
# --------------------------------------------------

svm_model = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

y_pred = svm_model.predict(X_test)

# Probability scores for AUC
y_prob = svm_model.predict_proba(X_test)


# --------------------------------------------------
# 6. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 7. TP, TN, FP, FN
#    One-vs-Rest for each class
# --------------------------------------------------

print("\nTP, TN, FP, FN for each class:")

for i, class_name in enumerate(iris.target_names):

    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    print(f"\nClass: {class_name}")
    print("TP =", TP)
    print("TN =", TN)
    print("FP =", FP)
    print("FN =", FN)


# --------------------------------------------------
# 8. Overall Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# Error rate
error = 1 - accuracy


# --------------------------------------------------
# 9. Specificity
#    Calculate specificity for each class
#    and take weighted average
# --------------------------------------------------

specificities = []
class_support = []

for i in range(len(iris.target_names)):

    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    specificity = TN / (TN + FP)

    specificities.append(specificity)
    class_support.append((y_test == i).sum())

specificity_weighted = np.average(
    specificities,
    weights=class_support
)


# --------------------------------------------------
# 10. AUC
#    Multi-class One-vs-Rest
# --------------------------------------------------

auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="weighted"
)


# --------------------------------------------------
# 11. Display Results
# --------------------------------------------------

print("\n======================================")
print("       SVM CLASSIFICATION RESULTS")
print("======================================")

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"Specificity   : {specificity_weighted:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"AUC           : {auc:.4f}")
print(f"Error Rate    : {error:.4f}")

print("======================================")
