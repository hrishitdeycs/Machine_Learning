```python
# ANN Classification using Wine Dataset
# 80% Training - 20% Testing

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# 1. Load Dataset
wine = load_wine()
X, y = wine.data, wine.target

# 2. Split Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# 3. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create and Train ANN
ann = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

ann.fit(X_train, y_train)

# 5. Make Predictions
y_pred = ann.predict(X_test)
y_prob = ann.predict_proba(X_test)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# 7. TP, TN, FP, FN
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)

print("\n========== CLASS-WISE RESULTS ==========")
for i, name in enumerate(wine.target_names):
    print(f"\n{name}:")
    print("  True Positive :", TP[i])
    print("  True Negative :", TN[i])
    print("  False Positive:", FP[i])
    print("  False Negative:", FN[i])

# 8. Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
specificity = np.mean(TN / (TN + FP))
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
error = 1 - accuracy

# 9. AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc = roc_auc_score(
    y_test_bin, y_prob, multi_class="ovr", average="macro"
)

# 10. Display Performance
print("\n========== PERFORMANCE ==========")
print(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision   : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall      : {recall:.4f} ({recall*100:.2f}%)")
print(f"Specificity : {specificity:.4f} ({specificity*100:.2f}%)")
print(f"F1-Score    : {f1:.4f} ({f1*100:.2f}%)")
print(f"Error       : {error:.4f} ({error*100:.2f}%)")
print(f"AUC         : {auc:.4f}")

# 11. Class-Wise Metrics Table
class_results = pd.DataFrame({
    "Class": wine.target_names,
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "Recall": TP / (TP + FN),
    "Specificity": TN / (TN + FP)
})

print("\n========== CLASS-WISE METRICS ==========")
print(class_results.to_string(index=False))

```
<img width="300" height="374" alt="image" src="https://github.com/user-attachments/assets/7ed411ee-1ea5-43e3-9ff9-dfb097dec22c" />
<img width="383" height="369" alt="image" src="https://github.com/user-attachments/assets/d936fddb-41bb-4415-981f-fe72f2242585" />
