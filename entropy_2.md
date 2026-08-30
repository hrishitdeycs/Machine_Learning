```python
# Decision Tree Classification using Entropy
# Dataset: loan.csv
# Train-Test Split: 80% Training / 20% Testing

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

df = pd.read_csv("07_loan_approval.csv")


# ---------------------------------------------------------
# 2. Convert Target Variable Y/N to 1/0
# ---------------------------------------------------------

# Y = Loan Approved = 1
# N = Loan Not Approved = 0

df["LoanApproved"] = df["LoanApproved"].map({
    "Y": 1,
    "N": 0
})


# ---------------------------------------------------------
# 3. Define X and y
# ---------------------------------------------------------

# Remove ApplicantID because it is only an identifier
X = df.drop(columns=["ApplicantID", "LoanApproved"])

y = df["LoanApproved"]

# ---------------------------------------------------------
# 4. Convert Categorical Variables
# ---------------------------------------------------------

# Columns such as Education, MaritalStatus,
# PropertyArea and SelfEmployed are categorical.

X = pd.get_dummies(X, drop_first=True)

# ---------------------------------------------------------
# 5. Handle Missing Values
# ---------------------------------------------------------

# Fill missing numerical values with median
for column in X.columns:
    if X[column].isnull().any():
        X[column] = X[column].fillna(X[column].median())

# ---------------------------------------------------------
# 6. Split Dataset: 80% Training, 20% Testing
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# ---------------------------------------------------------
# 7. Create Decision Tree using Entropy
# ---------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 8. Prediction
# ---------------------------------------------------------

y_pred = model.predict(X_test)

# Probability for Loan Approved = 1
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------
# 10. Calculate Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

error = 1 - accuracy

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)

# ---------------------------------------------------------
# 11. Display Confusion Matrix
# ---------------------------------------------------------

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print("                Predicted")
print("              N        Y")
print(f"Actual N    {TN:5d}    {FP:5d}")
print(f"Actual Y    {FN:5d}    {TP:5d}")

# ---------------------------------------------------------
# 12. Display Performance Metrics
# ---------------------------------------------------------

print("\n==========================================")
print("DECISION TREE - ENTROPY RESULTS")
print("==========================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-Score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

print("==========================================")

```
<img width="445" height="229" alt="image" src="https://github.com/user-attachments/assets/602c3b8d-ef11-46ed-bba1-c91853a6e50f" />
<img width="445" height="330" alt="image" src="https://github.com/user-attachments/assets/1f341f1a-d0d5-4544-b375-76bb94d08d7a" />
