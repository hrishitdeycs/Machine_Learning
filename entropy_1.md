```python
# ============================================================
# Decision Tree Classification using Entropy
# Train-Test Split: 80% - 20%
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("11_mushroom_edibility.csv")

# ============================================================
# 2. Define Features and Target
# ============================================================

# SampleID is only an identifier, so we remove it.
X = df.drop(columns=["SampleID", "Class"])
y = df["Class"]

# ============================================================
# 3. Encode Target Variable
# ============================================================

# Convert Class into 0 and 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClass Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(class_name, "=", i)

# ============================================================
# 4. Identify Categorical Columns
# ============================================================

categorical_columns = X.columns.tolist()

# ============================================================
# 5. One-Hot Encoding
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# Transform X
X_encoded = preprocessor.fit_transform(X)

# ============================================================
# 6. Split Dataset into 80% Training and 20% Testing
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    train_size=0.80,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ============================================================
# 7. Create Decision Tree using Entropy
# ============================================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree training completed.")

# ============================================================
# 8. Make Predictions
# ============================================================

y_pred = model.predict(X_test)

# Probability of positive class
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# 9. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Extract TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()

# ============================================================
# 10. Calculate Performance Metrics
# ============================================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Error
error = 1 - accuracy

# Recall / Sensitivity
recall = recall_score(y_test, y_pred)

# Specificity
specificity = TN / (TN + FP)

# F1 Score
f1 = f1_score(y_test, y_pred)

# AUC
auc = roc_auc_score(y_test, y_prob)

# ============================================================
# 11. Display Results
# ============================================================

print("\n================================================")
print("      DECISION TREE - ENTROPY RESULTS")
print("================================================")

print(f"Accuracy     : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"TP           : {TP}")
print(f"TN           : {TN}")
print(f"FP           : {FP}")
print(f"FN           : {FN}")
print(f"Error        : {error:.4f} ({error * 100:.2f}%)")
print(f"Recall       : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Specificity  : {specificity:.4f} ({specificity * 100:.2f}%)")
print(f"F1-score     : {f1:.4f}")
print(f"AUC          : {auc:.4f}")

print("================================================")


```
<img width="417" height="268" alt="image" src="https://github.com/user-attachments/assets/0d92b195-38a3-44e7-bd09-4a518ca0edde" />
<img width="531" height="328" alt="image" src="https://github.com/user-attachments/assets/753614e3-c762-41cd-87a8-1fdbdd028c79" />
