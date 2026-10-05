```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, confusion_matrix,
    recall_score, f1_score, roc_auc_score
)

# Load TXT dataset
df = pd.read_csv(
    "data_banknote_authentication.txt",
    header=None,
    names=["variance", "skewness", "curtosis", "entropy", "class"]
)

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM
# Removed probability=True
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Use decision_function instead of predict_proba()
# for ROC-AUC
y_score = model.decision_function(X_test)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
error = 1 - accuracy
auc = roc_auc_score(y_test, y_score)

# Results
print("\n===== LINEAR SVM =====")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"TP          : {tp}")
print(f"TN          : {tn}")
print(f"FP          : {fp}")
print(f"FN          : {fn}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

# Normal Confusion Matrix
print("\nConfusion Matrix")
print("                 Predicted")
print("                 0       1")
print(f"Actual 0       {tn:4d}    {fp:4d}")
print(f"Actual 1       {fn:4d}    {tp:4d}")
```
<img width="469" height="409" alt="image" src="https://github.com/user-attachments/assets/b412aa92-b741-4155-83fa-75c8a4d7d84c" />
