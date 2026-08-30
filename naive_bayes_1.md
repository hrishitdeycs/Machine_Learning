```python
# Naive Bayes classification for mushroom edibility

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
file_path = "11_mushroom_edibility.csv"

df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------
# SampleID is an identifier, so it is excluded.
X = df.drop(columns=["SampleID", "Class"])
y = df["Class"]

# Remove rows with missing values
data = pd.concat([X, y], axis=1).dropna()

X = data.drop(columns=["Class"])
y = data["Class"]

# ---------------------------------------------------------
# 3. Encode categorical variables
# ---------------------------------------------------------
# Encode every categorical feature into integers
X_encoded = X.copy()

feature_encoders = {}

for column in X_encoded.columns:
    encoder = LabelEncoder()
    X_encoded[column] = encoder.fit_transform(X_encoded[column].astype(str))
    feature_encoders[column] = encoder

# Encode target Class
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y.astype(str))

print("\nClass labels:")
for i, label in enumerate(target_encoder.classes_):
    print(i, "=", label)

# ---------------------------------------------------------
# 4. Split data: 80% training, 20% testing
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------------------------
# 5. Train Naive Bayes model
# ---------------------------------------------------------
model = CategoricalNB()

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Predictions
# ---------------------------------------------------------
y_pred = model.predict(X_test)

# Probability of positive class
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 7. Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Make sure this is a binary classification problem
if cm.shape != (2, 2):
    raise ValueError(
        "This code expects exactly two classes in the Class column."
    )

TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------
# 8. Performance metrics
# ---------------------------------------------------------

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Error rate
error = 1 - accuracy

# Recall / Sensitivity
recall = recall_score(y_test, y_pred, zero_division=0)

# Specificity
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

# F1-score
f1 = f1_score(y_test, y_pred, zero_division=0)

# AUC
auc = roc_auc_score(y_test, y_prob)

# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

print("\n========== NAIVE BAYES RESULTS ==========")

print(f"Accuracy    : {accuracy:.4f}")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

# ---------------------------------------------------------
# 10. Percentage results
# ---------------------------------------------------------

print("\n========== PERCENTAGE RESULTS ==========")

print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Error       : {error * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"Specificity : {specificity * 100:.2f}%")
print(f"F1-score    : {f1 * 100:.2f}%")
print(f"AUC         : {auc:.4f}")

```
<img width="297" height="223" alt="image" src="https://github.com/user-attachments/assets/ef1238da-5859-4208-b88c-a16894202cda" />

<img width="444" height="438" alt="image" src="https://github.com/user-attachments/assets/35f4d548-835b-48e2-b39f-9adbbc34736e" />

