```python
# KNN Classification on Iris Dataset

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_score
)

# --------------------------------------------------
# 1. Load Iris Dataset
# --------------------------------------------------
iris = load_iris()

X = iris.data
y = iris.target


# --------------------------------------------------
# 2. Split Dataset: 80% Training, 20% Testing
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
# 4. Create KNN Model
# --------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)


# --------------------------------------------------
# 5. Make Predictions
# --------------------------------------------------
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)


# --------------------------------------------------
# 6. Accuracy and Error
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy

# --------------------------------------------------
# 7. Confusion Matrix
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")

# Create labeled confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {name}" for name in iris.target_names],
    columns=[f"Predicted {name}" for name in iris.target_names]
)

print(cm_df)


# --------------------------------------------------
# 8. TP, TN, FP, FN for Each Class
# --------------------------------------------------
print("\n========== CLASS-WISE METRICS ==========")

total = np.sum(cm)

for i, class_name in enumerate(iris.target_names):

    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = total - (TP + FN + FP)

    precision = TP / (TP + FP)
    recall_class = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_class = 2 * (precision * recall_class) / (precision + recall_class)

    print(f"\n----- Class: {class_name} -----")
    print("TP          :", TP)
    print("TN          :", TN)
    print("FP          :", FP)
    print("FN          :", FN)
    print("Precision   :", precision)
    print("Recall      :", recall_class)
    print("Specificity :", specificity)
    print("F1-score    :", f1_class)


# --------------------------------------------------
# 9. Overall Recall
# --------------------------------------------------
recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n========== OVERALL METRICS ==========")
print("Accuracy    :", accuracy)
print("Error       :", error)
print("Recall      :", recall)


# --------------------------------------------------
# 10. Overall Specificity
# --------------------------------------------------
specificities = []

for i in range(len(iris.target_names)):

    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = total - (TP + FN + FP)

    specificity = TN / (TN + FP)
    specificities.append(specificity)

specificity_macro = np.mean(specificities)

print("Specificity :", specificity_macro)


# --------------------------------------------------
# 11. Overall F1 Score
# --------------------------------------------------
f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("F1-score    :", f1)


# --------------------------------------------------
# 12. AUC
# --------------------------------------------------
# Convert multiclass labels to binary format
y_test_binary = label_binarize(
    y_test,
    classes=[0, 1, 2]
)

auc = roc_auc_score(
    y_test_binary,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print("AUC         :", auc)


```
<img width="736" height="430" alt="image" src="https://github.com/user-attachments/assets/9883a4bb-c0e4-4a13-800c-eb470e801381" />
<img width="400" height="422" alt="image" src="https://github.com/user-attachments/assets/f7cacbd1-3017-4252-b1c1-d9b48b02ddb0" />
<img width="422" height="181" alt="image" src="https://github.com/user-attachments/assets/7a70799e-6131-43da-84a5-404fa8f52e5a" />
